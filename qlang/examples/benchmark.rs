//! Benchmark: Compare Interpreter vs JIT vs SIMD execution.
//!
//! Shows the performance progression:
//!   Interpreter (Rust) → JIT (LLVM scalar) → SIMD (LLVM AVX2)
//!
//! This demonstrates why QLANG compiles to machine code
//! instead of interpreting graphs like Python.

use std::time::Instant;

fn main() {
    println!("=== QLANG Performance Benchmark ===\n");

    use inkwell::context::Context;
    use inkwell::OptimizationLevel;
    use qlang_core::tensor::{Dtype, Shape, TensorType as TT};

    let sizes = [1024, 4096, 16384, 65536, 262144];

    println!("{:>10} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "Elements", "Interpreter", "JIT Scalar", "JIT SIMD", "JIT/Int", "SIMD/Int");
    println!("{}", "-".repeat(88));

    for &n in &sizes {
        // Build graph: y = relu(a + b)
        let mut e = qlang_agent::emitter::GraphEmitter::new("bench");
        let a = e.input("a", Dtype::F32, Shape::vector(n));
        let b = e.input("b", Dtype::F32, Shape::vector(n));
        let sum = e.add(a, b, TT::f32_vector(n));
        let activated = e.relu(sum, TT::f32_vector(n));
        e.output("y", activated, TT::f32_vector(n));
        let graph = e.build();

        let input_a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - (n as f32 * 0.005)).collect();
        let input_b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.005)).collect();

        // === Interpreter ===
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("a".into(), qlang_core::tensor::TensorData::from_f32(Shape::vector(n), &input_a));
        inputs.insert("b".into(), qlang_core::tensor::TensorData::from_f32(Shape::vector(n), &input_b));

        let start = Instant::now();
        let _interp = qlang_runtime::executor::execute(&graph, inputs).unwrap();
        let interp_time = start.elapsed();

        // === JIT Scalar ===
        let context = Context::create();
        let compiled = qlang_compile::codegen::compile_graph(&context, &graph, OptimizationLevel::Aggressive).unwrap();

        let start = Instant::now();
        let _jit_result = qlang_compile::codegen::execute_compiled(&compiled, &input_a, &input_b).unwrap();
        let jit_time = start.elapsed();

        // === JIT SIMD ===
        let simd_compiled = qlang_compile::simd::compile_graph_simd(&context, &graph).unwrap();

        let start = Instant::now();
        let _simd_result = qlang_compile::codegen::execute_compiled(&simd_compiled, &input_a, &input_b).unwrap();
        let simd_time = start.elapsed();

        let jit_speedup = interp_time.as_nanos() as f64 / jit_time.as_nanos().max(1) as f64;
        let simd_speedup = interp_time.as_nanos() as f64 / simd_time.as_nanos().max(1) as f64;

        println!("{n:>10} {interp_time:>14.2?} {jit_time:>14.2?} {simd_time:>14.2?} {jit_speedup:>9.1}x {simd_speedup:>9.1}x");
    }

    println!("\n=== Legend ===");
    println!("  Interpreter: Pure Rust, no LLVM (like Python)");
    println!("  JIT Scalar:  LLVM compiled, scalar float ops");
    println!("  JIT SIMD:    LLVM compiled, <8 x float> AVX2 vector ops");
    println!("  JIT/Int:     Speedup of JIT over Interpreter");
    println!("  SIMD/Int:    Speedup of SIMD over Interpreter");
}
