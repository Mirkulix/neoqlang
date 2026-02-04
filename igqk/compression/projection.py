"""
Optimal Projection for Compression

This module implements optimal projection onto compression submanifolds.

Definition 3.2 (Optimal Projection):
Π: M → N is optimal projection:
Π(θ) = argmin_{θ' ∈ N} d_M(θ, θ')

where d_M is the Riemannian distance on M.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..core.quantum_state import QuantumState
from ..core.measurement import ProjectiveMeasurement


class OptimalProjection:
    """
    Optimal projection onto compression submanifolds.

    Attributes:
        submanifold_type: Type of compression ('ternary', 'lowrank', 'sparse')
        compression_params: Parameters for compression
    """

    def __init__(
        self,
        submanifold_type: str = 'ternary',
        **compression_params
    ):
        """
        Initialize optimal projection.

        Args:
            submanifold_type: Compression type
            **compression_params: Additional parameters (rank, sparsity, etc.)
        """
        self.submanifold_type = submanifold_type
        self.compression_params = compression_params
        self.projector = ProjectiveMeasurement(submanifold_type, **compression_params)

    def project_state(self, rho: QuantumState) -> torch.Tensor:
        """
        Project quantum state onto compression submanifold.

        Args:
            rho: Quantum state

        Returns:
            Compressed weights θ* ∈ N
        """
        # Get mean parameter
        theta = rho.get_mean_parameter()

        # Project
        theta_compressed = self.projector.project(theta)

        return theta_compressed

    def project_model(self, model: nn.Module) -> nn.Module:
        """
        Project entire neural network model.

        Args:
            model: PyTorch model

        Returns:
            Compressed model
        """
        with torch.no_grad():
            for param in model.parameters():
                original_shape = param.shape
                flat_param = param.flatten()

                # Project
                compressed = self.projector.project(flat_param)

                # Reshape back
                param.data = compressed.view(original_shape)

        return model

    def compression_ratio(self, original_size: int) -> float:
        """
        Compute theoretical compression ratio.

        Args:
            original_size: Original parameter count

        Returns:
            Compression ratio (< 1 means compression)
        """
        if self.submanifold_type == 'ternary':
            # Ternary: 3 values → log2(3) ≈ 1.58 bits per weight
            # vs 32 bits float → ratio = 1.58/32 ≈ 0.05
            return 2.0 / 32.0  # ~1/16

        elif self.submanifold_type == 'lowrank':
            rank = self.compression_params.get('rank', 10)
            # Matrix m×n → rank decomposition: (m+n)×rank
            # Assume square matrix
            dim = int(np.sqrt(original_size))
            compressed_size = 2 * dim * rank
            return compressed_size / original_size

        elif self.submanifold_type == 'sparse':
            sparsity = self.compression_params.get('sparsity', 0.1)
            return sparsity

        else:
            return 1.0

    def distortion(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor
    ) -> float:
        """
        Compute compression distortion.

        D = ||θ - θ_compressed||²

        Args:
            original: Original weights
            compressed: Compressed weights

        Returns:
            Distortion
        """
        return torch.norm(original - compressed).item() ** 2
