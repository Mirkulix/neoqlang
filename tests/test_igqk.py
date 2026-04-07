"""
Quick test of IGQK system functionality.

Tests:
1. Import all modules
2. Create simple network
3. Initialize IGQK components
4. Test basic operations
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 80)
print("IGQK System Test")
print("=" * 80)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from igqk import (
        StatisticalManifold,
        QuantumState,
        QuantumGradientFlow,
        MeasurementOperator,
        IGQKOptimizer,
        IGQKTrainer,
        OptimalProjection
    )
    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}")
    exit(1)

# Test 2: Create simple network
print("\n[2/6] Creating simple neural network...")
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TinyNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"  [OK] Created TinyNet with {total_params} parameters")

# Test 3: Statistical manifold
print("\n[3/6] Testing Statistical Manifold...")
try:
    manifold = StatisticalManifold(model)
    theta = manifold.get_parameters()
    print(f"  [OK] Manifold dimension: {manifold.dim}")
    print(f"  [OK] Parameter vector shape: {theta.shape}")
except Exception as e:
    print(f"  [ERROR] Manifold failed: {e}")
    exit(1)

# Test 4: Quantum state
print("\n[4/6] Testing Quantum State...")
try:
    # Create quantum state from parameters
    rho = QuantumState.from_point(theta, rank=3)
    print(f"  [OK] Quantum state created")
    print(f"    - Dimension: {rho.dim}")
    print(f"    - Rank: {rho.rank}")
    print(f"    - Trace: {rho.trace():.4f}")
    print(f"    - Purity: {rho.purity():.4f}")
    print(f"    - Entropy: {rho.entropy():.4f}")
except Exception as e:
    print(f"  [ERROR] Quantum state failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Measurement
print("\n[5/6] Testing Measurement Operator...")
try:
    measurement = MeasurementOperator(weight_values=[-1.0, 0.0, 1.0])
    discrete_weights = measurement.measure(rho, method='threshold')
    unique_values = torch.unique(discrete_weights)
    print(f"  [OK] Measured quantum state")
    print(f"    - Discrete values: {unique_values.tolist()}")
    print(f"    - Weight distribution: {[(v.item(), (discrete_weights == v).sum().item()) for v in unique_values]}")
except Exception as e:
    print(f"  [ERROR] Measurement failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Compression
print("\n[6/6] Testing Optimal Projection...")
try:
    projection = OptimalProjection(submanifold_type='ternary')
    compressed = projection.project_state(rho)

    compression_ratio = projection.compression_ratio(rho.dim)
    print(f"  [OK] Compressed weights to ternary")
    print(f"    - Unique values: {torch.unique(compressed).tolist()}")
    print(f"    - Compression ratio: {compression_ratio:.4f}")

    # Compute distortion
    mean_param = rho.get_mean_parameter()
    distortion = projection.distortion(mean_param, compressed)
    print(f"    - Distortion: {distortion:.4f}")
except Exception as e:
    print(f"  [ERROR] Projection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 80)
print("[OK] All tests passed!")
print("=" * 80)
print("\nIGQK System is ready to use.")
print("\nNext steps:")
print("  1. Run MNIST example: python examples/mnist_example.py")
print("  2. See README.md for API documentation")
print("  3. Check mathematical details in PDFs")
print("\n" + "=" * 80)
