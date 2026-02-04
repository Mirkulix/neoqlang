"""
Measurement and Collapse Implementation

This module implements measurement operators for collapsing quantum states
to discrete weights.

Definition 4.1 (Weight Measurement):
A measurement operator on M is a family {M_w : w ∈ W} with:
1. M_w ≥ 0 for all w
2. Σ_w M_w = I (completeness)

where W is the space of discrete weights (e.g., W = {-1, 0, +1}^{m×n}).

Born Rule:
P(w | ρ) = Tr(ρ M_w)
"""

import numpy as np
import torch
from typing import List, Optional, Callable
from .quantum_state import QuantumState


class MeasurementOperator:
    """
    Measurement operator for collapsing quantum states to discrete weights.

    Attributes:
        weight_values: Possible discrete weight values (e.g., [-1, 0, 1])
        measurement_basis: Basis for measurements
    """

    def __init__(self, weight_values: List[float] = [-1.0, 0.0, 1.0]):
        """
        Initialize measurement operator.

        Args:
            weight_values: Discrete weight values (default: ternary)
        """
        self.weight_values = weight_values
        self.num_values = len(weight_values)

    def measure(
        self,
        rho: QuantumState,
        method: str = 'threshold'
    ) -> torch.Tensor:
        """
        Measure quantum state and collapse to discrete weights.

        P(w | ρ) = Tr(ρ M_w)

        Args:
            rho: Quantum state to measure
            method: Measurement method ('threshold', 'sample', 'optimal')

        Returns:
            Discrete weight vector [n]
        """
        # Get mean parameter
        mean_param = rho.get_mean_parameter()

        if method == 'threshold':
            return self._threshold_measure(mean_param)
        elif method == 'sample':
            return self._sample_measure(rho)
        elif method == 'optimal':
            return self._optimal_measure(rho)
        else:
            raise ValueError(f"Unknown measurement method: {method}")

    def _threshold_measure(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Threshold-based measurement for ternary weights.

        Maps θ_i to:
        - +1 if θ_i > threshold_high
        - -1 if θ_i < threshold_low
        - 0 otherwise

        Args:
            theta: Continuous parameter vector

        Returns:
            Discrete weight vector
        """
        if len(self.weight_values) == 3:  # Ternary: {-1, 0, +1}
            # Adaptive thresholds based on std
            std = theta.std()
            threshold_high = 0.5 * std
            threshold_low = -0.5 * std

            discrete = torch.zeros_like(theta)
            discrete[theta > threshold_high] = 1.0
            discrete[theta < threshold_low] = -1.0

        elif len(self.weight_values) == 2:  # Binary: {-1, +1}
            discrete = torch.sign(theta)
            discrete[discrete == 0] = 1.0  # Handle zeros

        else:
            # General case: quantize to nearest value
            discrete = torch.zeros_like(theta)
            for val in self.weight_values:
                mask = (theta - val).abs() == min(
                    [(theta - v).abs() for v in self.weight_values]
                )
                discrete[mask] = val

        return discrete

    def _sample_measure(self, rho: QuantumState) -> torch.Tensor:
        """
        Sample-based measurement using Born rule.

        Sample eigenstates proportionally to P(i) = λ_i,
        then quantize.

        Args:
            rho: Quantum state

        Returns:
            Discrete weight vector
        """
        # Sample eigenstate
        sampled = rho.sample(num_samples=1).squeeze()

        # Quantize
        return self._threshold_measure(sampled)

    def _optimal_measure(self, rho: QuantumState) -> torch.Tensor:
        """
        Optimal measurement minimizing expected distortion.

        For each parameter θ_i, choose discrete value that minimizes:
        E_ρ[(θ_i - w_i)²]

        Args:
            rho: Quantum state

        Returns:
            Discrete weight vector
        """
        # Get mean and variance for each parameter
        mean = rho.get_mean_parameter()

        # For each parameter, choose nearest discrete value
        discrete = torch.zeros_like(mean)

        for i in range(mean.shape[0]):
            # Find closest discrete value
            distances = [(mean[i] - val).abs().item() for val in self.weight_values]
            min_idx = distances.index(min(distances))
            discrete[i] = self.weight_values[min_idx]

        return discrete

    def probability(self, rho: QuantumState, weight_config: torch.Tensor) -> float:
        """
        Compute probability of measuring specific weight configuration.

        P(w | ρ) = Tr(ρ M_w)

        Args:
            rho: Quantum state
            weight_config: Discrete weight configuration [n]

        Returns:
            Probability
        """
        # Approximate: P(w) ≈ exp(-β ||E[ρ] - w||²)
        # where E[ρ] is mean parameter

        mean = rho.get_mean_parameter()
        distance = torch.norm(mean - weight_config) ** 2

        # Boltzmann-like probability with inverse temperature
        beta = 1.0
        prob = torch.exp(-beta * distance).item()

        return prob

    def fidelity_to_target(
        self,
        rho: QuantumState,
        target_weights: torch.Tensor
    ) -> float:
        """
        Compute fidelity between quantum state and target discrete weights.

        Args:
            rho: Quantum state
            target_weights: Target discrete weights

        Returns:
            Fidelity (higher = better)
        """
        measured = self.measure(rho, method='optimal')
        return 1.0 / (1.0 + torch.norm(measured - target_weights).item())


class ProjectiveMeasurement:
    """
    Projective measurement onto compression submanifold.

    Projects quantum state onto discrete weight submanifold N ⊂ M.
    """

    def __init__(
        self,
        submanifold_type: str = 'ternary',
        rank: Optional[int] = None,
        sparsity: Optional[float] = None
    ):
        """
        Initialize projective measurement.

        Args:
            submanifold_type: Type of submanifold ('ternary', 'lowrank', 'sparse')
            rank: Rank for low-rank projection
            sparsity: Sparsity level for sparse projection
        """
        self.submanifold_type = submanifold_type
        self.rank = rank
        self.sparsity = sparsity

    def project(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Project continuous parameter onto compression submanifold.

        Π: M → N is optimal projection:
        Π(θ) = argmin_{θ' ∈ N} d_M(θ, θ')

        Args:
            theta: Continuous parameter vector

        Returns:
            Projected parameter on submanifold N
        """
        if self.submanifold_type == 'ternary':
            return self._project_ternary(theta)
        elif self.submanifold_type == 'lowrank':
            return self._project_lowrank(theta)
        elif self.submanifold_type == 'sparse':
            return self._project_sparse(theta)
        else:
            raise ValueError(f"Unknown submanifold type: {self.submanifold_type}")

    def _project_ternary(self, theta: torch.Tensor) -> torch.Tensor:
        """Project onto ternary weight space {-1, 0, +1}."""
        # Adaptive thresholds
        std = theta.std()
        threshold = 0.5 * std

        projected = torch.zeros_like(theta)
        projected[theta > threshold] = 1.0
        projected[theta < -threshold] = -1.0

        return projected

    def _project_lowrank(self, theta: torch.Tensor) -> torch.Tensor:
        """Project onto low-rank subspace."""
        if self.rank is None:
            raise ValueError("Rank must be specified for low-rank projection")

        # Reshape to matrix (assume square for simplicity)
        dim = int(np.sqrt(theta.shape[0]))
        if dim * dim != theta.shape[0]:
            # Not square, use truncated SVD on vector
            return theta  # TODO: Implement properly

        matrix = theta.view(dim, dim)

        # SVD
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # Truncate to rank
        S_trunc = torch.zeros_like(S)
        S_trunc[:self.rank] = S[:self.rank]

        # Reconstruct
        matrix_lowrank = torch.mm(torch.mm(U, torch.diag(S_trunc)), Vh)

        return matrix_lowrank.flatten()

    def _project_sparse(self, theta: torch.Tensor) -> torch.Tensor:
        """Project onto sparse weight space."""
        if self.sparsity is None:
            raise ValueError("Sparsity must be specified for sparse projection")

        # Keep only top-k largest magnitude weights
        k = int(self.sparsity * theta.shape[0])

        # Get top-k indices
        _, indices = torch.topk(theta.abs(), k)

        # Zero out all others
        projected = torch.zeros_like(theta)
        projected[indices] = theta[indices]

        return projected

    def optimal_measurement_operator(
        self,
        rho: QuantumState,
        target_submanifold: str
    ) -> torch.Tensor:
        """
        Construct optimal measurement operator for projection onto submanifold.

        Theorem 4.1: M_w = |ψ_w⟩⟨ψ_w|
        where |ψ_w⟩ is eigenstate of ρ closest to w ∈ N.

        Args:
            rho: Quantum state
            target_submanifold: Target submanifold type

        Returns:
            Optimal measurement operator M_w
        """
        # Get eigenstates
        eigenvectors = rho.eigenvectors

        # Project each eigenstate onto target submanifold
        projected_states = []
        for i in range(rho.rank):
            eigenstate = eigenvectors[:, i]
            projected = self.project(eigenstate)
            projected_states.append(projected)

        projected_states = torch.stack(projected_states, dim=1)

        # M_w = projectors onto these states
        # For simplicity, return projected states
        return projected_states
