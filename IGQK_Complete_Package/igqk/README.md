# IGQK: Information Geometric Quantum Compression

A PyTorch-based implementation of Information Geometric Quantum Compression for efficient neural network training and compression.

## Overview

IGQK is a unified theoretical framework that combines:
- **Information Geometry**: Statistical manifolds with Fisher metric
- **Quantum Mechanics**: Density matrices and quantum gradient flow
- **Riemannian Geometry**: Optimal projection for compression

## Key Features

- 🎯 **Unified Framework**: Combines quantization, pruning, and low-rank compression
- 📊 **Theoretical Guarantees**: Convergence proofs and rate-distortion bounds
- 🚀 **Efficient Implementation**: GPU-accelerated with PyTorch
- 🔬 **Research-Ready**: Modular design for experimentation

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from igqk import IGQKOptimizer, TernaryManifold

# Define your model
model = YourNeuralNetwork()

# Create IGQK optimizer
optimizer = IGQKOptimizer(
    model.parameters(),
    manifold=TernaryManifold(),
    hbar=0.1,  # Quantum uncertainty
    gamma=0.01  # Damping
)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Compress to ternary weights
compressed_model = optimizer.compress()
```

## Architecture

```
igqk/
├── core/           # Core quantum gradient flow
├── manifolds/      # Statistical manifolds (Fisher metric)
├── compression/    # Projection algorithms
├── optimizers/     # IGQK optimizer variants
└── utils/          # Utilities and metrics
```

## Theory

Based on the paper "Information Geometric Quantum Compression: A Unified Theory for Efficient Neural Networks" (Manus AI, 2026).

### Key Theorems

1. **Convergence**: Quantum gradient flow converges to ε-optimal state
2. **Rate-Distortion**: Fundamental compression bound D ≥ O((n-k)²/n)
3. **Entanglement**: Correlated weights improve generalization

## Examples

See `examples/` for:
- MNIST classification with ternary weights
- CIFAR-10 compression benchmark
- Transformer compression

## Citation

```bibtex
@article{manus2026igqk,
  title={Information Geometric Quantum Compression: A Unified Theory for Efficient Neural Networks},
  author={Manus AI},
  year={2026}
}
```

## License

MIT License
