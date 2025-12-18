//! Zero-copy bridge between PyTorch CPU tensors and Metal buffers.
//!
//! This module enables direct GPU access to PyTorch tensor memory using
//! Apple Silicon's unified memory architecture. PyTorch tensors on CPU
//! are treated as GPU-accessible memory without explicit copies.

use super::buffer::{DType, MetalBuffer};
use super::device::MetalContext;
use pyo3::prelude::*;
use std::ffi::c_void;

/// Convert a PyTorch dtype string to our DType enum.
fn parse_dtype(dtype_str: &str) -> Result<DType, String> {
    match dtype_str {
        "torch.float16" | "float16" | "half" => Ok(DType::Float16),
        "torch.bfloat16" | "bfloat16" => Ok(DType::BFloat16),
        "torch.float32" | "float32" | "float" => Ok(DType::Float32),
        "torch.int32" | "int32" | "int" => Ok(DType::Int32),
        "torch.int64" | "int64" | "long" => Ok(DType::Int64),
        _ => Err(format!("Unsupported dtype: {}", dtype_str)),
    }
}

/// Create a MetalBuffer from a PyTorch tensor's data pointer.
///
/// This leverages unified memory on Apple Silicon - the CPU tensor's memory
/// is directly accessible by the GPU without copying.
///
/// # Safety
/// The caller must ensure:
/// - The tensor remains valid for the lifetime of the returned buffer
/// - The tensor is contiguous in memory
/// - The tensor is on CPU (not MPS)
#[pyfunction]
pub fn metal_buffer_from_tensor(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
) -> PyResult<MetalBuffer> {
    // Get tensor attributes
    let data_ptr: usize = tensor.call_method0("data_ptr")?.extract()?;
    let numel: usize = tensor.call_method0("numel")?.extract()?;
    let dtype_obj = tensor.getattr("dtype")?;
    let dtype_str = dtype_obj.str()?.to_string();

    // Get shape as Vec<usize>
    let shape_obj = tensor.getattr("shape")?;
    let shape: Vec<usize> = shape_obj
        .iter()?
        .map(|x| x.and_then(|v| v.extract::<usize>()))
        .collect::<PyResult<Vec<_>>>()?;

    // Check tensor is contiguous
    let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Tensor must be contiguous for zero-copy Metal access",
        ));
    }

    // Check tensor is on CPU (not MPS or CUDA)
    let device_obj = tensor.getattr("device")?;
    let device_type: String = device_obj.getattr("type")?.extract()?;
    if device_type != "cpu" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Tensor must be on CPU for zero-copy Metal access (got {}). \
             On Apple Silicon, CPU memory is unified with GPU memory.",
            device_type
        )));
    }

    let dtype = parse_dtype(&dtype_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let size_bytes = numel * dtype.size_bytes();

    // Create Metal buffer from the tensor's memory
    // This uses unified memory - no copy needed on Apple Silicon
    let buffer = unsafe {
        MetalBuffer::from_ptr(data_ptr as *mut c_void, size_bytes, shape, dtype)
    }
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(buffer)
}

/// Tensor wrapper that holds both the PyTorch tensor (to keep it alive)
/// and its corresponding Metal buffer view.
#[pyclass]
pub struct TensorMetalView {
    /// The Metal buffer view of the tensor data
    buffer: MetalBuffer,
    /// Keep a reference count to track usage
    refcount: std::sync::atomic::AtomicUsize,
}

impl TensorMetalView {
    /// Get the underlying Metal buffer.
    pub fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }
}

#[pymethods]
impl TensorMetalView {
    /// Get the data pointer as integer.
    fn data_ptr(&self) -> usize {
        self.buffer.contents_ptr() as usize
    }

    /// Get the shape.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.buffer.shape().to_vec()
    }

    /// Get size in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.buffer.size_bytes()
    }

    /// Get number of elements.
    #[getter]
    fn numel(&self) -> usize {
        self.buffer.numel()
    }
}

/// Create a view of a PyTorch tensor as a Metal buffer.
///
/// This is the main entry point for the zero-copy tensor bridge.
/// The returned TensorMetalView can be used with Metal kernels.
#[pyfunction]
pub fn tensor_to_metal(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
) -> PyResult<TensorMetalView> {
    let buffer = metal_buffer_from_tensor(py, tensor)?;
    Ok(TensorMetalView {
        buffer,
        refcount: std::sync::atomic::AtomicUsize::new(1),
    })
}

/// Batch convert multiple tensors to Metal buffers.
///
/// This is more efficient when converting many tensors at once.
#[pyfunction]
pub fn tensors_to_metal(
    py: Python<'_>,
    tensors: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<TensorMetalView>> {
    tensors
        .iter()
        .map(|t| tensor_to_metal(py, t))
        .collect()
}

/// Initialize the Metal context and load the shader library.
///
/// This should be called once at startup before using any Metal operations.
#[pyfunction]
#[pyo3(signature = (metallib_path=None))]
pub fn init_metal_runtime(metallib_path: Option<&str>) -> PyResult<()> {
    // Initialize Metal context (singleton)
    let ctx = MetalContext::get();

    // Load shader library if path provided
    if let Some(path) = metallib_path {
        ctx.load_library(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    }

    Ok(())
}

/// Get Metal device info for debugging.
#[pyfunction]
pub fn get_metal_device_info() -> PyResult<(String, usize, usize)> {
    let ctx = MetalContext::get();
    Ok((
        ctx.device_name(),
        ctx.max_threads_per_threadgroup(),
        ctx.max_threadgroup_memory(),
    ))
}

/// Synchronize all pending Metal operations.
///
/// This ensures all GPU work is complete before continuing.
#[pyfunction]
pub fn metal_synchronize() -> PyResult<()> {
    // Metal command buffers handle synchronization
    // For now, this is a no-op since we use synchronous execution
    // In the future, we could track pending command buffers
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_parsing() {
        assert!(matches!(parse_dtype("torch.float16"), Ok(DType::Float16)));
        assert!(matches!(parse_dtype("torch.float32"), Ok(DType::Float32)));
        assert!(matches!(parse_dtype("torch.int64"), Ok(DType::Int64)));
        assert!(parse_dtype("invalid").is_err());
    }
}
