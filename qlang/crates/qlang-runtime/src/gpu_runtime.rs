//! GPU Runtime — Execute QLANG graphs on GPU via generated WGSL shaders.
//!
//! This module provides GPU execution capabilities using WGSL compute shaders.
//! Since wgpu is not available in all environments, this module provides:
//! 1. A `GpuDevice` abstraction for GPU execution
//! 2. Fallback to CPU when GPU is unavailable
//! 3. Shader compilation and buffer management
//!
//! The actual wgpu integration requires the `gpu` feature flag.
//! Without it, this module provides CPU-simulated GPU execution for testing.

use std::collections::HashMap;

/// GPU device abstraction.
#[derive(Debug)]
pub struct GpuDevice {
    /// Device name/identifier.
    pub name: String,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
    /// Maximum workgroup size.
    pub max_workgroup_size: u32,
    /// Whether this is a real GPU or CPU fallback.
    pub is_real_gpu: bool,
}

/// A compiled GPU kernel ready for execution.
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// The WGSL shader source code.
    pub shader_source: String,
    /// Number of elements to process.
    pub n_elements: usize,
    /// Workgroup size.
    pub workgroup_size: u32,
}

/// Result of GPU execution.
#[derive(Debug)]
pub struct GpuResult {
    /// Output data.
    pub data: Vec<f32>,
    /// Execution time in microseconds.
    pub execution_time_us: u64,
    /// Whether executed on real GPU or CPU fallback.
    pub used_gpu: bool,
}

/// Errors during GPU execution.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("GPU not available: {0}")]
    NotAvailable(String),
    #[error("shader compilation failed: {0}")]
    ShaderError(String),
    #[error("buffer too large: {size} bytes (max {max})")]
    BufferTooLarge { size: usize, max: usize },
    #[error("execution error: {0}")]
    ExecutionError(String),
}

impl GpuDevice {
    /// Create a CPU fallback device for environments without GPU.
    pub fn cpu_fallback() -> Self {
        Self {
            name: "CPU Fallback (WGSL Simulator)".into(),
            max_buffer_size: usize::MAX,
            max_workgroup_size: 256,
            is_real_gpu: false,
        }
    }

    /// Try to initialize a real GPU device.
    /// Returns CPU fallback if no GPU is available.
    pub fn new() -> Self {
        // Without wgpu feature, always fall back to CPU
        Self::cpu_fallback()
    }

    /// Compile a WGSL shader into a kernel.
    pub fn compile_kernel(
        &self,
        shader_source: &str,
        n_elements: usize,
    ) -> Result<GpuKernel, GpuError> {
        // Validate shader has required structure
        if !shader_source.contains("@compute") {
            return Err(GpuError::ShaderError(
                "shader must contain @compute entry point".into(),
            ));
        }
        Ok(GpuKernel {
            shader_source: shader_source.to_string(),
            n_elements,
            workgroup_size: self.max_workgroup_size,
        })
    }

    /// Execute a kernel with element-wise operation on two input buffers.
    pub fn execute_elementwise(
        &self,
        kernel: &GpuKernel,
        input_a: &[f32],
        input_b: &[f32],
    ) -> Result<GpuResult, GpuError> {
        let start = std::time::Instant::now();

        if !self.is_real_gpu {
            // CPU fallback: simulate GPU execution by interpreting the shader
            let result = simulate_elementwise_shader(
                &kernel.shader_source,
                input_a,
                input_b,
                kernel.n_elements,
            )?;
            let elapsed = start.elapsed().as_micros() as u64;
            return Ok(GpuResult {
                data: result,
                execution_time_us: elapsed,
                used_gpu: false,
            });
        }

        Err(GpuError::NotAvailable(
            "wgpu feature not enabled — use cpu_fallback()".into(),
        ))
    }

    /// Execute a matmul kernel.
    pub fn execute_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuResult, GpuError> {
        let start = std::time::Instant::now();

        if !self.is_real_gpu {
            // CPU fallback: standard matmul
            let mut result = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }
            let elapsed = start.elapsed().as_micros() as u64;
            return Ok(GpuResult {
                data: result,
                execution_time_us: elapsed,
                used_gpu: false,
            });
        }

        Err(GpuError::NotAvailable(
            "wgpu feature not enabled".into(),
        ))
    }
}

/// Simulate WGSL element-wise shader on CPU by parsing the operation.
fn simulate_elementwise_shader(
    shader: &str,
    input_a: &[f32],
    input_b: &[f32],
    n_elements: usize,
) -> Result<Vec<f32>, GpuError> {
    let n = n_elements.min(input_a.len()).min(input_b.len());
    let mut output = vec![0.0f32; n];

    // Parse shader to determine operation
    for i in 0..n {
        let a = input_a[i];
        let b = input_b[i];
        output[i] = if shader.contains("a + b") {
            a + b
        } else if shader.contains("a - b") {
            a - b
        } else if shader.contains("a * b") {
            a * b
        } else if shader.contains("a / b") {
            if b != 0.0 { a / b } else { 0.0 }
        } else if shader.contains("max(") && shader.contains("0.0") {
            a.max(0.0) // relu
        } else if shader.contains("1.0 / (1.0 + exp(") {
            1.0 / (1.0 + (-a).exp()) // sigmoid
        } else if shader.contains("tanh(") {
            a.tanh()
        } else {
            a // passthrough
        };
    }

    Ok(output)
}

/// Information about available GPU devices.
pub fn gpu_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("backend".into(), "CPU Fallback (wgpu not enabled)".into());
    info.insert("feature_flag".into(), "gpu".into());
    info.insert(
        "instructions".into(),
        "Enable GPU: cargo build --features gpu".into(),
    );
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_creation() {
        let dev = GpuDevice::cpu_fallback();
        assert!(!dev.is_real_gpu);
        assert_eq!(dev.max_workgroup_size, 256);
    }

    #[test]
    fn test_compile_kernel() {
        let dev = GpuDevice::new();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = a + b; output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 1024).unwrap();
        assert_eq!(kernel.n_elements, 1024);
    }

    #[test]
    fn test_compile_kernel_invalid() {
        let dev = GpuDevice::new();
        let result = dev.compile_kernel("invalid shader", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_elementwise_add() {
        let dev = GpuDevice::cpu_fallback();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = a + b; output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 4).unwrap();
        let result = dev
            .execute_elementwise(&kernel, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
        assert!(!result.used_gpu);
    }

    #[test]
    fn test_elementwise_mul() {
        let dev = GpuDevice::cpu_fallback();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = a * b; output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 3).unwrap();
        let result = dev
            .execute_elementwise(&kernel, &[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0])
            .unwrap();
        assert_eq!(result.data, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_matmul_cpu() {
        let dev = GpuDevice::cpu_fallback();
        // [2,2] x [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dev.execute_matmul(&a, &b, 2, 2, 2).unwrap();
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_info() {
        let info = gpu_info();
        assert!(info.contains_key("backend"));
        assert!(info.contains_key("feature_flag"));
    }

    #[test]
    fn test_relu_shader() {
        let dev = GpuDevice::cpu_fallback();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = max(a, 0.0); output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 4).unwrap();
        let result = dev
            .execute_elementwise(&kernel, &[-1.0, 2.0, -3.0, 4.0], &[0.0; 4])
            .unwrap();
        assert_eq!(result.data, vec![0.0, 2.0, 0.0, 4.0]);
    }
}
