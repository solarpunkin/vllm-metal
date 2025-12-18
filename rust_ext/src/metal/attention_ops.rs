//! Python-exposed attention operations using Metal kernels.
//!
//! These functions provide the interface between Python/PyTorch and
//! the Metal attention kernels, handling tensor conversion and dispatch.

use super::buffer::{DType, MetalBuffer};
use super::device::MetalContext;
use super::dispatch::{dispatch_paged_attention, dispatch_sdpa};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::path::Path;

/// Ensure the Metal shader library is loaded.
fn ensure_library_loaded() -> PyResult<()> {
    let ctx = MetalContext::get();
    if ctx.library().is_some() {
        return Ok(());
    }

    // Try to load from the build output directory
    let metallib_path = option_env!("METALLIB_PATH");
    if let Some(path) = metallib_path {
        if Path::new(path).exists() {
            ctx.load_library(Path::new(path))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            return Ok(());
        }
    }

    // Try common locations
    let candidates = [
        "target/release/build/vllm_metal_rust-*/out/vllm_kernels.metallib",
        "target/debug/build/vllm_metal_rust-*/out/vllm_kernels.metallib",
        "rust_ext/vllm_kernels.metallib",
        "vllm_kernels.metallib",
    ];

    for pattern in candidates {
        if let Ok(paths) = glob::glob(pattern) {
            for path in paths.flatten() {
                if path.exists() {
                    ctx.load_library(&path)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                    return Ok(());
                }
            }
        }
    }

    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "Metal shader library not found. Build the Rust extension first.",
    ))
}

/// Helper to create a MetalBuffer from tensor attributes.
fn tensor_to_buffer(
    data_ptr: usize,
    shape: Vec<usize>,
    dtype_str: &str,
) -> PyResult<MetalBuffer> {
    let dtype = match dtype_str {
        "torch.float16" | "float16" | "half" => DType::Float16,
        "torch.bfloat16" | "bfloat16" => DType::BFloat16,
        "torch.float32" | "float32" | "float" => DType::Float32,
        "torch.int32" | "int32" | "int" => DType::Int32,
        "torch.int64" | "int64" | "long" => DType::Int64,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype: {}",
                dtype_str
            )))
        }
    };

    let numel: usize = shape.iter().product();
    let size_bytes = numel * dtype.size_bytes();

    unsafe { MetalBuffer::from_ptr(data_ptr as *mut c_void, size_bytes, shape, dtype) }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Scaled dot-product attention using Metal kernels.
///
/// This is the main attention operation for prefill and decode.
///
/// Args:
///     query: Query tensor [num_queries, num_heads, head_dim] on CPU
///     key: Key tensor [num_queries, seq_len, num_kv_heads, head_dim] on CPU
///     value: Value tensor [num_queries, seq_len, num_kv_heads, head_dim] on CPU
///     output: Output tensor [num_queries, num_heads, head_dim] on CPU
///     scale: Attention scale factor
#[pyfunction]
pub fn metal_sdpa(
    py: Python<'_>,
    query: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    scale: f32,
) -> PyResult<()> {
    ensure_library_loaded()?;

    // Extract tensor info
    let q_ptr: usize = query.call_method0("data_ptr")?.extract()?;
    let k_ptr: usize = key.call_method0("data_ptr")?.extract()?;
    let v_ptr: usize = value.call_method0("data_ptr")?.extract()?;
    let out_ptr: usize = output.call_method0("data_ptr")?.extract()?;

    let q_shape: Vec<usize> = query
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let k_shape: Vec<usize> = key
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = query.getattr("dtype")?.str()?.to_string();

    // Create Metal buffers
    let q_buf = tensor_to_buffer(q_ptr, q_shape.clone(), &dtype_str)?;
    let k_buf = tensor_to_buffer(k_ptr, k_shape.clone(), &dtype_str)?;
    let v_buf = tensor_to_buffer(
        v_ptr,
        k_shape.clone(), // V has same shape as K
        &dtype_str,
    )?;
    let out_buf = tensor_to_buffer(out_ptr, q_shape.clone(), &dtype_str)?;

    // Extract dimensions
    let num_queries = q_shape[0] as i32;
    let num_heads = q_shape[1] as i32;
    let head_dim = q_shape[2] as i32;
    let seq_len = k_shape[1] as i32;
    let num_kv_heads = k_shape[2] as i32;

    // Dispatch Metal kernel
    dispatch_sdpa(
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        num_queries,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        scale,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Paged attention for decode phase with KV cache.
///
/// This implements vLLM's paged attention algorithm using Metal kernels.
///
/// Args:
///     query: Query tensor [batch, num_heads, head_dim] on CPU
///     key_cache: KV cache keys [num_blocks, block_size, num_kv_heads, head_dim]
///     value_cache: KV cache values [num_blocks, block_size, num_kv_heads, head_dim]
///     block_tables: Block table [batch, max_blocks_per_seq]
///     seq_lens: Sequence lengths [batch]
///     output: Output tensor [batch, num_heads, head_dim]
///     scale: Attention scale factor
///     block_size: Size of each cache block
#[pyfunction]
#[pyo3(signature = (query, key_cache, value_cache, block_tables, seq_lens, output, scale, block_size=16))]
pub fn metal_paged_attention(
    py: Python<'_>,
    query: &Bound<'_, PyAny>,
    key_cache: &Bound<'_, PyAny>,
    value_cache: &Bound<'_, PyAny>,
    block_tables: &Bound<'_, PyAny>,
    seq_lens: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    scale: f32,
    block_size: i32,
) -> PyResult<()> {
    ensure_library_loaded()?;

    // Extract tensor info
    let q_ptr: usize = query.call_method0("data_ptr")?.extract()?;
    let kc_ptr: usize = key_cache.call_method0("data_ptr")?.extract()?;
    let vc_ptr: usize = value_cache.call_method0("data_ptr")?.extract()?;
    let bt_ptr: usize = block_tables.call_method0("data_ptr")?.extract()?;
    let sl_ptr: usize = seq_lens.call_method0("data_ptr")?.extract()?;
    let out_ptr: usize = output.call_method0("data_ptr")?.extract()?;

    let q_shape: Vec<usize> = query
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let kc_shape: Vec<usize> = key_cache
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let bt_shape: Vec<usize> = block_tables
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = query.getattr("dtype")?.str()?.to_string();

    // Create Metal buffers
    let q_buf = tensor_to_buffer(q_ptr, q_shape.clone(), &dtype_str)?;
    let kc_buf = tensor_to_buffer(kc_ptr, kc_shape.clone(), &dtype_str)?;
    let vc_buf = tensor_to_buffer(vc_ptr, kc_shape.clone(), &dtype_str)?;
    let bt_buf = tensor_to_buffer(bt_ptr, bt_shape.clone(), "int32")?;

    let sl_shape = vec![q_shape[0]];
    let sl_buf = tensor_to_buffer(sl_ptr, sl_shape, "int32")?;

    let out_buf = tensor_to_buffer(out_ptr, q_shape.clone(), &dtype_str)?;

    // Extract dimensions
    let batch_size = q_shape[0] as i32;
    let num_heads = q_shape[1] as i32;
    let head_dim = q_shape[2] as i32;
    let num_blocks = kc_shape[0] as i32;
    let num_kv_heads = kc_shape[2] as i32;
    let max_blocks_per_seq = bt_shape[1] as i32;

    // Dispatch Metal kernel
    dispatch_paged_attention(
        &q_buf,
        &kc_buf,
        &vc_buf,
        &bt_buf,
        &sl_buf,
        &out_buf,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        num_blocks,
        scale,
        max_blocks_per_seq,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Load the Metal shader library from a specific path.
#[pyfunction]
pub fn load_metal_library(path: &str) -> PyResult<()> {
    let ctx = MetalContext::get();
    ctx.load_library(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Check if Metal is available and working.
#[pyfunction]
pub fn is_metal_available() -> bool {
    // Try to get the Metal context - if this succeeds, Metal is available
    let ctx = MetalContext::get();
    !ctx.device_name().is_empty()
}

/// Get information about the Metal device.
#[pyfunction]
pub fn metal_device_info() -> PyResult<(String, usize, usize)> {
    let ctx = MetalContext::get();
    Ok((
        ctx.device_name(),
        ctx.max_threads_per_threadgroup(),
        ctx.max_threadgroup_memory(),
    ))
}
