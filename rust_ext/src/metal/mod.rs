//! Metal integration module for vLLM on Apple Silicon.
//!
//! Provides direct Metal GPU access for high-performance inference:
//! - Device singleton for GPU management
//! - Zero-copy buffer management with unified memory
//! - Compute pipeline caching
//! - Custom kernel dispatch
//! - PyTorch tensor bridge for zero-copy GPU access

pub mod device;
pub mod buffer;
pub mod pipeline;
pub mod kernels;
pub mod dispatch;
pub mod tensor_bridge;
pub mod attention_ops;

pub use device::{MetalContext, PyMetalContext};
pub use buffer::MetalBuffer;
pub use pipeline::ComputePipeline;
pub use dispatch::{dispatch_sdpa, dispatch_paged_attention};
pub use tensor_bridge::{TensorMetalView, tensor_to_metal, init_metal_runtime};
pub use attention_ops::{metal_sdpa, metal_paged_attention, is_metal_available};
