//! vLLM Metal Rust Extensions
//!
//! High-performance Rust implementations for vLLM on Apple Silicon (MPS).
//! This module provides GPU Model Runner V2-style optimizations:
//! - Persistent batch state management
//! - Vectorized block table operations
//! - Fast input preparation pipeline
//! - Zero-copy tensor conversions
//! - Custom Metal kernels for attention, GEMV, RoPE, RMS norm

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

mod batch_state;
mod block_table;
mod input_prep;

// Metal integration module (macOS only)
#[cfg(target_os = "macos")]
pub mod metal;

// ============================================================================
// Tensor Conversion Functions (Original)
// ============================================================================

/// Fast conversion of numpy array to nested Python list.
#[pyfunction]
fn tensor_to_nested_list(py: Python<'_>, arr: PyReadonlyArray2<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];

    let outer_list = PyList::empty_bound(py);

    for i in 0..rows {
        let inner_list = PyList::empty_bound(py);
        for j in 0..cols {
            inner_list.append(arr[[i, j]])?;
        }
        outer_list.append(inner_list)?;
    }

    Ok(outer_list.into())
}

/// Fast conversion of 1D numpy array to Python list of lists (each with 1 element).
#[pyfunction]
fn tensor_1d_to_nested_list(py: Python<'_>, arr: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let len = arr.len();

    let outer_list = PyList::empty_bound(py);

    for i in 0..len {
        let inner_list = PyList::empty_bound(py);
        inner_list.append(arr[i])?;
        outer_list.append(inner_list)?;
    }

    Ok(outer_list.into())
}

/// Fast conversion of flat numpy array to Python list.
#[pyfunction]
fn tensor_to_flat_list(py: Python<'_>, arr: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let arr = arr.as_array();
    let len = arr.len();

    let list = PyList::empty_bound(py);

    for i in 0..len {
        list.append(arr[i])?;
    }

    Ok(list.into())
}

/// Fast comparison of request ID lists.
#[pyfunction]
fn compare_req_ids(ids_a: Vec<String>, ids_b: Vec<String>) -> bool {
    if ids_a.len() != ids_b.len() {
        return false;
    }
    ids_a.iter().zip(ids_b.iter()).all(|(a, b)| a == b)
}

// ============================================================================
// Python Module Definition
// ============================================================================

#[pymodule]
fn vllm_metal_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Original tensor conversion functions
    m.add_function(wrap_pyfunction!(tensor_to_nested_list, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_1d_to_nested_list, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_flat_list, m)?)?;
    m.add_function(wrap_pyfunction!(compare_req_ids, m)?)?;

    // V2 Batch State Manager
    m.add_class::<batch_state::BatchStateManager>()?;

    // V2 Block Table Manager
    m.add_class::<block_table::BlockTableManager>()?;

    // V2 Input Preparation
    m.add_function(wrap_pyfunction!(input_prep::prepare_decode_inputs_v2, m)?)?;
    m.add_function(wrap_pyfunction!(input_prep::prepare_prefill_inputs_v2, m)?)?;
    m.add_function(wrap_pyfunction!(input_prep::compute_slot_mapping_batch, m)?)?;
    m.add_function(wrap_pyfunction!(input_prep::build_attn_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(input_prep::compute_logits_indices, m)?)?;
    m.add_function(wrap_pyfunction!(input_prep::compute_req_indices, m)?)?;

    // Block table utilities
    m.add_function(wrap_pyfunction!(block_table::compute_slot_mapping_vectorized, m)?)?;

    // Metal classes and functions (macOS only)
    #[cfg(target_os = "macos")]
    {
        // Classes
        m.add_class::<metal::buffer::MetalBuffer>()?;
        m.add_class::<metal::device::PyMetalContext>()?;
        m.add_class::<metal::tensor_bridge::TensorMetalView>()?;

        // Tensor bridge functions
        m.add_function(wrap_pyfunction!(metal::tensor_bridge::metal_buffer_from_tensor, m)?)?;
        m.add_function(wrap_pyfunction!(metal::tensor_bridge::tensor_to_metal, m)?)?;
        m.add_function(wrap_pyfunction!(metal::tensor_bridge::tensors_to_metal, m)?)?;
        m.add_function(wrap_pyfunction!(metal::tensor_bridge::init_metal_runtime, m)?)?;
        m.add_function(wrap_pyfunction!(metal::tensor_bridge::get_metal_device_info, m)?)?;
        m.add_function(wrap_pyfunction!(metal::tensor_bridge::metal_synchronize, m)?)?;

        // Attention operations
        m.add_function(wrap_pyfunction!(metal::attention_ops::metal_sdpa, m)?)?;
        m.add_function(wrap_pyfunction!(metal::attention_ops::metal_paged_attention, m)?)?;
        m.add_function(wrap_pyfunction!(metal::attention_ops::load_metal_library, m)?)?;
        m.add_function(wrap_pyfunction!(metal::attention_ops::is_metal_available, m)?)?;
        m.add_function(wrap_pyfunction!(metal::attention_ops::metal_device_info, m)?)?;
    }

    Ok(())
}
