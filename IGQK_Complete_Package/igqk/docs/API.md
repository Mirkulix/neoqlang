# IGQK API Reference

## Core Components

### `QuantumState`

Represents a quantum state (density matrix) on a statistical manifold.

```python
from igqk import QuantumState

state = QuantumState(eigenvectors, eigenvalues, device=None)
```

**Methods:**

- `expectation(observable)`: Compute expectation value Tr(ρ O)
- `von_neumann_entropy()`: Compute entropy S(ρ) = -Tr(ρ log ρ)
- `purity()`: Compute purity Tr(ρ²)
- `to_classical()`: Collapse to classical parameter vector
- `sample(n_samples)`: Sample classical states from quantum distribution
- `from_classical(params, hbar)`: Create quantum state from classical parameters

### `QuantumGradientFlow`

Implements quantum gradient flow dynamics.

```python
from igqk import QuantumGradientFlow

qgf = QuantumGradientFlow(hbar=0.1, gamma=0.01, mass=1.0)
new_state = qgf.step(state, gradient, fisher_metric=None, dt=0.01)
```

## Manifolds

### `StatisticalManifold` (Abstract Base Class)

Base class for statistical manifolds with Fisher metric.

**Methods:**

- `fisher_metric(model, data, target)`: Compute Fisher information matrix
- `geodesic_distance(params1, params2)`: Compute geodesic distance
- `natural_gradient(gradient, fisher, damping)`: Compute natural gradient

### `DiagonalFisherManifold`

Diagonal approximation of Fisher information (efficient).

```python
from igqk import DiagonalFisherManifold

manifold = DiagonalFisherManifold(n_samples=100)
fisher = manifold.fisher_metric(model, data, target)
```

### `EmpiricalFisherManifold`

Full empirical Fisher information matrix.

```python
from igqk import EmpiricalFisherManifold

manifold = EmpiricalFisherManifold(n_samples=100)
```

### `BlockDiagonalFisherManifold`

Block-diagonal approximation (per-layer).

```python
from igqk import BlockDiagonalFisherManifold

manifold = BlockDiagonalFisherManifold(n_samples=100)
```

## Compression

### `TernaryProjector`

Projects weights to ternary values {-1, 0, +1}.

```python
from igqk import TernaryProjector

projector = TernaryProjector(method='optimal', threshold=0.3)
compressed = projector.project(params)
```

**Methods:**
- `'sign'`: Simple sign function
- `'threshold'`: Threshold-based (|w| < threshold → 0)
- `'optimal'`: Minimize L2 distortion

**Compression:** 16× (32-bit → 2-bit)

### `BinaryProjector`

Projects weights to binary values {-1, +1}.

```python
from igqk import BinaryProjector

projector = BinaryProjector()
```

**Compression:** 32× (32-bit → 1-bit)

### `SparseProjector`

Prunes weights (sets to zero).

```python
from igqk import SparseProjector

projector = SparseProjector(sparsity=0.9)  # 90% sparse
```

### `LowRankProjector`

Low-rank matrix approximation via SVD.

```python
from igqk import LowRankProjector

projector = LowRankProjector(rank=10)
# or
projector = LowRankProjector(rank_ratio=0.5)
```

### `HybridProjector`

Combines multiple compression techniques.

```python
from igqk import HybridProjector, SparseProjector, TernaryProjector

projector = HybridProjector([
    SparseProjector(sparsity=0.9),
    TernaryProjector(method='optimal')
])
```

### Utility Functions

```python
from igqk import compress_model, measure_compression

# Compress entire model
compressed_model = compress_model(model, projector, inplace=False)

# Measure compression statistics
stats = measure_compression(original_model, compressed_model)
# Returns: {
#   'original_params': int,
#   'compressed_params': int,
#   'original_memory_mb': float,
#   'compressed_memory_mb': float,
#   'compression_ratio': float,
#   'distortion': float
# }
```

## Optimizers

### `IGQKOptimizer`

Main IGQK optimizer implementing quantum gradient flow.

```python
from igqk import IGQKOptimizer

optimizer = IGQKOptimizer(
    params,
    lr=0.01,
    hbar=0.1,
    gamma=0.01,
    manifold=None,
    projector=None,
    use_quantum=True,
    ensemble_size=1
)
```

**Parameters:**

- `params`: Model parameters (from `model.parameters()`)
- `lr`: Learning rate (default: 0.01)
- `hbar`: Quantum uncertainty parameter (default: 0.1)
- `gamma`: Damping coefficient (default: 0.01)
- `manifold`: Statistical manifold (default: DiagonalFisherManifold)
- `projector`: Compression projector (default: TernaryProjector)
- `use_quantum`: Enable quantum dynamics (default: True)
- `ensemble_size`: Number of particles for ensemble (default: 1)

**Methods:**

- `step(closure=None)`: Perform optimization step
- `compress(model=None)`: Compress model parameters
- `get_quantum_state(param)`: Get quantum state for parameter
- `entropy()`: Compute total von Neumann entropy
- `purity()`: Compute average purity

### `IGQKScheduler`

Learning rate scheduler for IGQK.

```python
from igqk import IGQKScheduler

scheduler = IGQKScheduler(
    optimizer,
    mode='cosine',
    T_max=100,
    hbar_min=0.01,
    gamma_max=0.1
)

# In training loop
for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

**Parameters:**

- `optimizer`: IGQKOptimizer instance
- `mode`: Scheduling mode ('cosine', 'linear', 'exponential')
- `T_max`: Maximum number of steps
- `hbar_min`: Minimum quantum uncertainty
- `gamma_max`: Maximum damping

## Complete Example

```python
import torch
import torch.nn as nn
from igqk import (
    IGQKOptimizer,
    IGQKScheduler,
    TernaryProjector,
    DiagonalFisherManifold,
    measure_compression
)

# Model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Optimizer
optimizer = IGQKOptimizer(
    model.parameters(),
    lr=0.01,
    hbar=0.2,
    gamma=0.01,
    manifold=DiagonalFisherManifold(n_samples=100),
    projector=TernaryProjector(method='optimal'),
    use_quantum=True
)

# Scheduler
scheduler = IGQKScheduler(
    optimizer,
    mode='cosine',
    T_max=num_epochs
)

# Training
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Monitor quantum metrics
    print(f"Entropy: {optimizer.entropy():.4f}")
    print(f"Purity: {optimizer.purity():.4f}")

# Compress
original_model = copy.deepcopy(model)
optimizer.compress(model)

# Measure
stats = measure_compression(original_model, model)
print(f"Compression: {stats['compression_ratio']:.2f}×")
```
