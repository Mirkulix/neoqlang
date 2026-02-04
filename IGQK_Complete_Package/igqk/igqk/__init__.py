"""
IGQK: Information Geometric Quantum Compression

A unified framework for efficient neural network training and compression.
"""

__version__ = "0.1.0"

# Core components
from .core.quantum_state import QuantumState, QuantumGradientFlow

# Manifolds
from .manifolds.statistical_manifold import (
    StatisticalManifold,
    EmpiricalFisherManifold,
    DiagonalFisherManifold,
    BlockDiagonalFisherManifold,
)

# Compression
from .compression.projectors import (
    CompressionProjector,
    TernaryProjector,
    BinaryProjector,
    LowRankProjector,
    SparseProjector,
    HybridProjector,
    compress_model,
    measure_compression,
)

# Optimizers
from .optimizers.igqk_optimizer import IGQKOptimizer, IGQKScheduler

__all__ = [
    # Core
    "QuantumState",
    "QuantumGradientFlow",
    # Manifolds
    "StatisticalManifold",
    "EmpiricalFisherManifold",
    "DiagonalFisherManifold",
    "BlockDiagonalFisherManifold",
    # Compression
    "CompressionProjector",
    "TernaryProjector",
    "BinaryProjector",
    "LowRankProjector",
    "SparseProjector",
    "HybridProjector",
    "compress_model",
    "measure_compression",
    # Optimizers
    "IGQKOptimizer",
    "IGQKScheduler",
]
