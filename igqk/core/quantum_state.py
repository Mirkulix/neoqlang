"""
Quantum State Implementation (Density Matrices)

This module implements quantum states on the statistical manifold
as described in IGQK Theory.

Definition 1.2 (Quantum State on M):
A quantum state on M is a density matrix ρ: M → ℂ^{d×d} with:
1. ρ(θ) ≥ 0 (positive semidefinite) for all θ ∈ M
2. Tr(ρ(θ)) = 1 (normalized)
3. ρ is smooth as mapping M → ℂ^{d×d}

Interpretation: ρ(θ) describes a superposition of weight configurations around θ.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from scipy.linalg import sqrtm, logm, expm


class QuantumState:
    """
    Density matrix representation of quantum state.

    For efficiency, we use low-rank approximation:
    ρ = Σ_i λ_i |ψ_i⟩⟨ψ_i| where d << n

    Attributes:
        eigenvalues: λ_i (shape: [d]) - probabilities
        eigenvectors: |ψ_i⟩ (shape: [n, d]) - weight configurations
        dim: n - dimension of parameter space
        rank: d - rank of density matrix
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        check_properties: bool = True
    ):
        """
        Initialize quantum state from eigendecomposition.

        Args:
            eigenvalues: Eigenvalues λ_i (must be non-negative, sum to 1)
            eigenvectors: Eigenvectors |ψ_i⟩ (must be orthonormal)
            check_properties: If True, verify ρ ≥ 0 and Tr(ρ) = 1
        """
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.dim = eigenvectors.shape[0]
        self.rank = eigenvalues.shape[0]

        if check_properties:
            self._verify_properties()

    def _verify_properties(self):
        """Verify density matrix properties."""
        # Check positive semidefinite: λ_i ≥ 0
        if torch.any(self.eigenvalues < -1e-10):
            raise ValueError(f"Negative eigenvalues: {self.eigenvalues.min().item()}")

        # Clamp small negative values to 0
        self.eigenvalues = torch.clamp(self.eigenvalues, min=0.0)

        # Check normalized: Tr(ρ) = Σ λ_i = 1
        trace = self.eigenvalues.sum()
        if abs(trace - 1.0) > 1e-6:
            # Auto-normalize
            self.eigenvalues = self.eigenvalues / trace

        # Check orthonormal eigenvectors (optional, expensive)
        # V^T V should be identity
        if self.rank <= 100:  # Only check for small rank
            gram = torch.mm(self.eigenvectors.T, self.eigenvectors)
            identity = torch.eye(self.rank, device=gram.device)
            error = torch.norm(gram - identity)
            if error > 1e-3:
                print(f"Warning: Eigenvectors not orthonormal (error: {error:.6f})")

    @classmethod
    def from_point(cls, theta: torch.Tensor, rank: int = 1) -> 'QuantumState':
        """
        Create pure state |θ⟩⟨θ| from parameter vector.

        Args:
            theta: Parameter vector θ
            rank: Rank of state (1 = pure state)

        Returns:
            Quantum state ρ = |θ⟩⟨θ|
        """
        dim = theta.shape[0]
        if rank == 1:
            # Pure state
            eigenvalues = torch.ones(1, device=theta.device)
            eigenvectors = theta.unsqueeze(1) / torch.norm(theta)
        else:
            # Mixed state with small noise
            eigenvalues = torch.ones(rank, device=theta.device) / rank
            eigenvectors = torch.randn(dim, rank, device=theta.device)
            # Gram-Schmidt orthonormalization
            eigenvectors, _ = torch.linalg.qr(eigenvectors)

        return cls(eigenvalues, eigenvectors, check_properties=True)

    @classmethod
    def from_matrix(cls, rho: torch.Tensor, rank: Optional[int] = None) -> 'QuantumState':
        """
        Create quantum state from full density matrix.

        Args:
            rho: Full density matrix [n x n]
            rank: Keep only top-k eigenvalues (None = all)

        Returns:
            Quantum state with low-rank approximation
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(rho)

        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only positive eigenvalues
        positive_mask = eigenvalues > 1e-10
        eigenvalues = eigenvalues[positive_mask]
        eigenvectors = eigenvectors[:, positive_mask]

        # Truncate to rank
        if rank is not None and eigenvalues.shape[0] > rank:
            eigenvalues = eigenvalues[:rank]
            eigenvectors = eigenvectors[:, :rank]
            # Renormalize
            eigenvalues = eigenvalues / eigenvalues.sum()

        return cls(eigenvalues, eigenvectors, check_properties=True)

    def to_matrix(self) -> torch.Tensor:
        """
        Convert to full density matrix.

        ρ = Σ_i λ_i |ψ_i⟩⟨ψ_i| = V Λ V^T

        Returns:
            Full density matrix [n x n]
        """
        # ρ = V @ diag(λ) @ V^T
        V_scaled = self.eigenvectors * torch.sqrt(self.eigenvalues).unsqueeze(0)
        rho = torch.mm(V_scaled, V_scaled.T)
        return rho

    def expectation(self, observable: torch.Tensor) -> float:
        """
        Compute expectation value ⟨O⟩ = Tr(ρO).

        Args:
            observable: Observable operator O [n x n] or vector [n]

        Returns:
            Expectation value
        """
        if observable.dim() == 1:
            # For vector: ⟨θ⟩ = Tr(ρ ⊗ θ) = Σ_i λ_i ⟨ψ_i|θ⟩
            # This is just weighted sum
            projections = torch.mv(self.eigenvectors.T, observable)
            return torch.dot(self.eigenvalues, projections).item()
        else:
            # For matrix: Tr(ρO) = Tr(VΛV^T O) = Σ_i λ_i ⟨ψ_i|O|ψ_i⟩
            # O_psi = V^T @ O @ V (transform O to eigenbasis)
            O_psi = torch.mm(torch.mm(self.eigenvectors.T, observable), self.eigenvectors)
            # Trace in eigenbasis weighted by eigenvalues
            return torch.sum(self.eigenvalues * torch.diag(O_psi)).item()

    def entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ) = -Σ_i λ_i log λ_i.

        Returns:
            Entropy (measure of quantum uncertainty)
        """
        # Filter out zero eigenvalues
        positive_eigs = self.eigenvalues[self.eigenvalues > 1e-10]
        if len(positive_eigs) == 0:
            return 0.0
        return -torch.sum(positive_eigs * torch.log(positive_eigs)).item()

    def purity(self) -> float:
        """
        Compute purity Tr(ρ²) = Σ_i λ_i².

        Purity = 1 for pure states, < 1 for mixed states.

        Returns:
            Purity
        """
        return torch.sum(self.eigenvalues ** 2).item()

    def trace(self) -> float:
        """Compute trace Tr(ρ) = Σ_i λ_i (should be 1)."""
        return self.eigenvalues.sum().item()

    def fidelity(self, other: 'QuantumState') -> float:
        """
        Compute fidelity F(ρ, σ) = Tr(√(√ρ σ √ρ)).

        Args:
            other: Other quantum state σ

        Returns:
            Fidelity (1 = identical, 0 = orthogonal)
        """
        # For efficiency, use approximation for low-rank states
        # F ≈ Σ_i Σ_j √(λ_i μ_j) |⟨ψ_i|φ_j⟩|²
        overlaps = torch.mm(self.eigenvectors.T, other.eigenvectors)  # [rank1 x rank2]
        overlaps_sq = overlaps ** 2

        fidelity = 0.0
        for i in range(self.rank):
            for j in range(other.rank):
                fidelity += np.sqrt(self.eigenvalues[i].item() * other.eigenvalues[j].item()) * overlaps_sq[i, j].item()

        return fidelity

    def renormalize(self):
        """Renormalize to ensure Tr(ρ) = 1."""
        trace = self.eigenvalues.sum()
        if abs(trace - 1.0) > 1e-6:
            self.eigenvalues = self.eigenvalues / trace

    def get_mean_parameter(self) -> torch.Tensor:
        """
        Get mean parameter E_ρ[θ] = Σ_i λ_i ψ_i.

        Returns:
            Mean parameter vector [n]
        """
        # Weighted sum of eigenvectors
        return torch.mv(self.eigenvectors, self.eigenvalues)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample parameter vectors from quantum state.

        Args:
            num_samples: Number of samples

        Returns:
            Sampled parameter vectors [num_samples x n]
        """
        # Sample eigenstate according to Born rule P(i) = λ_i
        indices = torch.multinomial(self.eigenvalues, num_samples, replacement=True)

        # Return corresponding eigenvectors
        return self.eigenvectors[:, indices].T

    def truncate_rank(self, new_rank: int):
        """
        Truncate to lower rank (keep top eigenvalues).

        Args:
            new_rank: New rank (must be ≤ current rank)
        """
        if new_rank >= self.rank:
            return

        self.eigenvalues = self.eigenvalues[:new_rank]
        self.eigenvectors = self.eigenvectors[:, :new_rank]
        self.rank = new_rank

        # Renormalize
        self.renormalize()

    def __repr__(self) -> str:
        return (f"QuantumState(dim={self.dim}, rank={self.rank}, "
                f"trace={self.trace():.4f}, purity={self.purity():.4f}, "
                f"entropy={self.entropy():.4f})")
