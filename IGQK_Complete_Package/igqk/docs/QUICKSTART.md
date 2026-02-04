# IGQK Quick Start Guide

## Installation

```bash
git clone https://github.com/manus-ai/igqk.git
cd igqk
pip install -e .
```

## Basic Usage

### 1. Import IGQK

```python
import torch
import torch.nn as nn
from igqk import IGQKOptimizer, TernaryProjector
```

### 2. Define Your Model

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### 3. Create IGQK Optimizer

```python
optimizer = IGQKOptimizer(
    model.parameters(),
    lr=0.01,           # Learning rate
    hbar=0.1,          # Quantum uncertainty
    gamma=0.01,        # Damping coefficient
    use_quantum=True,  # Enable quantum dynamics
    projector=TernaryProjector(method='optimal')
)
```

### 4. Training Loop

```python
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 5. Compress Model

```python
# Compress to ternary weights
optimizer.compress(model)

# Evaluate compressed model
test_accuracy = evaluate(model, test_loader)
print(f"Compressed accuracy: {test_accuracy:.2f}%")
```

## Advanced Features

### Quantum Metrics

Monitor quantum properties during training:

```python
# Von Neumann entropy
entropy = optimizer.entropy()

# Purity (1 = pure state, <1 = mixed)
purity = optimizer.purity()

print(f"Entropy: {entropy:.4f}, Purity: {purity:.4f}")
```

### Custom Compression

Use different compression strategies:

```python
from igqk import BinaryProjector, SparseProjector, HybridProjector

# Binary quantization
binary_proj = BinaryProjector()

# Sparse pruning (90% sparsity)
sparse_proj = SparseProjector(sparsity=0.9)

# Hybrid: sparse + ternary
hybrid_proj = HybridProjector([sparse_proj, TernaryProjector()])

optimizer = IGQKOptimizer(
    model.parameters(),
    projector=hybrid_proj
)
```

### Learning Rate Scheduling

Adjust quantum parameters during training:

```python
from igqk import IGQKScheduler

scheduler = IGQKScheduler(
    optimizer,
    mode='cosine',
    T_max=num_epochs,
    hbar_min=0.01,
    gamma_max=0.1
)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

### Measure Compression

```python
from igqk import measure_compression

stats = measure_compression(original_model, compressed_model)

print(f"Compression ratio: {stats['compression_ratio']:.2f}×")
print(f"Memory: {stats['original_memory_mb']:.2f} MB → {stats['compressed_memory_mb']:.2f} MB")
print(f"Distortion: {stats['distortion']:.4f}")
```

## Hyperparameter Guide

### Learning Rate (`lr`)
- **Range**: 0.001 - 0.1
- **Default**: 0.01
- **Effect**: Controls step size in parameter space

### Quantum Uncertainty (`hbar`)
- **Range**: 0.01 - 0.5
- **Default**: 0.1
- **Effect**: Higher values → more exploration, lower values → more exploitation
- **Tip**: Start high (0.2), decay to low (0.01)

### Damping (`gamma`)
- **Range**: 0.001 - 0.1
- **Default**: 0.01
- **Effect**: Higher values → faster convergence, lower values → more stable
- **Tip**: Start low (0.01), increase if training is slow

## Examples

See `examples/` directory for complete examples:
- `mnist_ternary.py`: MNIST classification with ternary compression
- `compression_benchmark.py`: Compare different compression methods

## Troubleshooting

### Training is unstable
- Reduce `hbar` (quantum uncertainty)
- Increase `gamma` (damping)
- Reduce learning rate

### Model doesn't compress well
- Try different projection methods: `'sign'`, `'threshold'`, `'optimal'`
- Use hybrid compression
- Train longer before compressing

### Out of memory
- Disable quantum dynamics: `use_quantum=False`
- Use diagonal Fisher approximation (default)
- Reduce batch size

## Next Steps

- Read the [full documentation](API.md)
- Check out the [theory paper](../IGQK_Paper.md)
- Run the [benchmark](../examples/compression_benchmark.py)
- Join the [discussion](https://github.com/manus-ai/igqk/discussions)
