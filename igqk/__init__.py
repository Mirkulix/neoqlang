"""
IGQK - Information-Geometric Quantum Compression

A theoretical framework for neural network compression combining:
- Information Geometry (Fisher metric on statistical manifolds)
- Quantum Mechanics (superposition and entanglement)
- Compression Theory (projection onto low-dimensional submanifolds)
"""

__version__ = '0.1.0'
__author__ = 'IGQK Research Team'

from .core.manifold import StatisticalManifold
from .core.quantum_state import QuantumState
from .core.evolution import QuantumGradientFlow
from .core.measurement import MeasurementOperator
from .integration.pytorch import IGQKOptimizer, IGQKTrainer
from .compression.projection import OptimalProjection

__all__ = [
    'StatisticalManifold',
    'QuantumState',
    'QuantumGradientFlow',
    'MeasurementOperator',
    'IGQKOptimizer',
    'IGQKTrainer',
    'OptimalProjection',
]
