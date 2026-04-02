# QLANG

**Graph-based AI-to-AI Programming Language**

QLANG is a programming language where programs are directed acyclic graphs (DAGs), not text. Designed for AI systems to communicate computations directly, without the lossy detour through human-readable syntax.

Built on **IGQK** (Information-Geometric Quantum Compression) theory.

## Quick Start

```bash
# Build
cd qlang
cargo build --workspace

# Run tests (90+)
cargo test --workspace

# Run examples
cargo run --example hello_qlang          # Simple graph
cargo run --example neural_network       # MLP + IGQK compression
cargo run --example jit_compile          # LLVM JIT compilation
cargo run --example train_autograd       # Backpropagation training
cargo run --example train_mnist          # MNIST (784->128->10)
cargo run --example transformer          # Transformer encoder
cargo run --example benchmark            # Performance comparison
cargo run --example full_pipeline        # Everything in one run

# CLI
cargo run --bin qlang-cli -p qlang-compile -- repl               # Interactive REPL
cargo run --bin qlang-cli -p qlang-compile -- parse file.qlang   # Parse .qlang text
cargo run --bin qlang-cli -p qlang-compile -- info file.qlg.json # Show graph info
cargo run --bin qlang-cli -p qlang-compile -- jit file.qlg.json  # JIT execute
cargo run --bin qlang-cli -p qlang-compile -- compile file.qlg.json -o out.o  # AOT compile
cargo run --bin qlang-cli -p qlang-compile -- asm file.qlg.json  # Show assembly
cargo run --bin qlang-cli -p qlang-compile -- dot file.qlg.json  # Graphviz output
```

## .qlang Syntax

```qlang
graph mnist_classifier {
  input x: f32[1, 784]
  input W1: f32[784, 128]
  input W2: f32[128, 10]

  node h = matmul(x, W1)
  node a = relu(h)
  node logits = matmul(a, W2)
  node probs = softmax(logits)

  // IGQK ternary compression with formal proof
  node compressed = to_ternary(W1) @proof theorem_5_2

  output predictions = probs
  output ternary_weights = compressed
}
```

## Architecture

```
qlang/
+-- crates/
|   +-- qlang-core/        Graph, Tensor, Quantum, Ops, Verify, Serial
|   +-- qlang-compile/     LLVM JIT, SIMD, AOT, WASM, GPU, Parser, REPL, LSP
|   +-- qlang-runtime/     Executor, Autograd, Training, Transformer, Profiler
|   +-- qlang-agent/       Emitter, Protocol, Compose, Packages, Distributed, Diff
+-- examples/              11 runnable examples
+-- spec/                  Language specification
```

## Compilation Targets

| Target | Format | Use Case |
|--------|--------|----------|
| Interpreter | Rust | Development, debugging |
| LLVM JIT | x86-64 native | Production (22x faster) |
| LLVM SIMD | AVX2 vectors | Batch processing |
| AOT | .o object file | Link with C/Rust/C++ |
| Assembly | .S | Inspection |
| WebAssembly | .wat/.wasm | Browser, Node.js |
| GPU | WGSL | WebGPU compute shaders |
| Binary | .qlg | Wire format (compact) |
| JSON | .json | Human-readable |

## Performance

```
Operation: relu(a + b), release mode, best of 10 runs

Elements    Interpreter    JIT Scalar     Speedup
   1,024         7.1us       721ns         9.8x
  64,000       721.0us        34us        21.4x
1,000,000      16.4ms       733us        22.4x
```

## ML Capabilities

- **Autograd**: Reverse-mode automatic differentiation (backpropagation)
- **Training**: SGD with cross-entropy loss, softmax output
- **MNIST**: 784->128->10 MLP, trains to 100% on synthetic data
- **Transformer**: Multi-head attention, LayerNorm, GELU, positional encoding
- **IGQK Compression**: 4-16x ternary compression with formal proof guarantees

## IGQK Theory

QLANG implements the IGQK training algorithm:

1. Initialize quantum state rho_0
2. Evolve via quantum gradient flow: d_rho/dt = -i[H, rho] - gamma{G^-1 nabla L, rho}
3. Project onto compression submanifold (ternary, low-rank, sparse)
4. Quantum measurement via Born rule: P(w|rho) = Tr(rho M_w)

Formal theorems from the IGQK paper are attached as proof annotations to compression operations.

## Requirements

- Rust 1.70+ (tested with 1.93)
- LLVM 18 development libraries (for JIT/AOT compilation)
- Optional: `libzstd-dev`, `libpolly-18-dev`

## License

MIT
