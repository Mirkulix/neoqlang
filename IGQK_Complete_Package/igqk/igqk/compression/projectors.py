"""
Projection algorithms for compressing neural networks.

Implements optimal projection onto compression submanifolds.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from abc import ABC, abstractmethod


class CompressionProjector(ABC):
    """
    Abstract base class for compression projectors.
    
    A projector maps parameters from the full manifold M to a
    compression submanifold N ⊂ M.
    """
    
    @abstractmethod
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """
        Project parameters onto compression submanifold.
        
        Args:
            params: Full-precision parameters
            
        Returns:
            Compressed parameters
        """
        pass
    
    @abstractmethod
    def compression_ratio(self) -> float:
        """
        Return the compression ratio achieved.
        """
        pass


class TernaryProjector(CompressionProjector):
    """
    Projects weights to ternary values {-1, 0, +1}.
    
    Achieves 16× compression (32-bit float → 2 bits).
    """
    
    def __init__(
        self,
        method: Literal['sign', 'threshold', 'optimal'] = 'optimal',
        threshold: float = 0.3
    ):
        """
        Args:
            method: Projection method
                - 'sign': Simple sign function
                - 'threshold': Threshold-based (|w| < threshold → 0)
                - 'optimal': Minimize distortion
            threshold: Threshold for 'threshold' method
        """
        self.method = method
        self.threshold = threshold
        
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """
        Project to ternary values.
        """
        if self.method == 'sign':
            return torch.sign(params)
        
        elif self.method == 'threshold':
            # Threshold small values to zero
            mask = torch.abs(params) >= self.threshold
            ternary = torch.sign(params) * mask.float()
            return ternary
        
        elif self.method == 'optimal':
            # Optimal ternary quantization minimizing L2 distortion
            # Based on: "Ternary Weight Networks" (Li et al., 2016)
            
            # Compute optimal threshold
            abs_params = torch.abs(params)
            threshold = 0.7 * torch.mean(abs_params)
            
            # Compute optimal scale
            mask = abs_params >= threshold
            if mask.sum() > 0:
                scale = torch.mean(abs_params[mask])
            else:
                scale = 1.0
            
            # Quantize
            ternary = torch.zeros_like(params)
            ternary[params > threshold] = scale
            ternary[params < -threshold] = -scale
            
            return ternary
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compression_ratio(self) -> float:
        """32-bit float to 2-bit ternary."""
        return 32.0 / 2.0  # 16×


class BinaryProjector(CompressionProjector):
    """
    Projects weights to binary values {-1, +1}.
    
    Achieves 32× compression (32-bit float → 1 bit).
    """
    
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """
        Project to binary values using sign function.
        """
        return torch.sign(params)
    
    def compression_ratio(self) -> float:
        """32-bit float to 1-bit binary."""
        return 32.0 / 1.0  # 32×


class LowRankProjector(CompressionProjector):
    """
    Projects weight matrices to low-rank approximation.
    
    Uses SVD: W ≈ U Σ Vᵀ with only top-k singular values.
    """
    
    def __init__(self, rank: Optional[int] = None, rank_ratio: float = 0.5):
        """
        Args:
            rank: Target rank (if None, use rank_ratio)
            rank_ratio: Fraction of original rank to keep
        """
        self.rank = rank
        self.rank_ratio = rank_ratio
        
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """
        Project to low-rank approximation.
        
        Args:
            params: Weight matrix (2D tensor)
            
        Returns:
            Low-rank approximation
        """
        if params.ndim != 2:
            # For non-matrix tensors, return as-is
            return params
        
        # Compute SVD
        U, S, Vt = torch.linalg.svd(params, full_matrices=False)
        
        # Determine rank
        if self.rank is not None:
            k = min(self.rank, len(S))
        else:
            k = max(1, int(len(S) * self.rank_ratio))
        
        # Reconstruct with top-k singular values
        low_rank = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        
        return low_rank
    
    def compression_ratio(self) -> float:
        """
        Compression depends on matrix dimensions and rank.
        For m×n matrix with rank k: (m*n) / (k*(m+n))
        """
        # Return approximate ratio
        return 1.0 / self.rank_ratio


class SparseProjector(CompressionProjector):
    """
    Projects weights to sparse representation (pruning).
    
    Keeps only top-k% of weights by magnitude.
    """
    
    def __init__(self, sparsity: float = 0.9):
        """
        Args:
            sparsity: Fraction of weights to set to zero (0.9 = 90% sparse)
        """
        assert 0 <= sparsity < 1
        self.sparsity = sparsity
        
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """
        Project to sparse representation.
        """
        # Compute threshold
        abs_params = torch.abs(params)
        k = max(1, int((1 - self.sparsity) * params.numel()))  # Ensure at least 1
        k = min(k, params.numel())  # Don't exceed total elements
        
        if k == 0:
            return torch.zeros_like(params)
        
        threshold = torch.topk(abs_params.flatten(), k).values[-1]
        
        # Mask small weights
        mask = abs_params >= threshold
        sparse = params * mask.float()
        
        return sparse
    
    def compression_ratio(self) -> float:
        """
        Sparse storage: only non-zero values + indices.
        Approximate: 1 / (1 - sparsity)
        """
        return 1.0 / (1.0 - self.sparsity)


class HybridProjector(CompressionProjector):
    """
    Combines multiple compression techniques.
    
    Example: Low-rank + ternary quantization
    """
    
    def __init__(self, projectors: list[CompressionProjector]):
        """
        Args:
            projectors: List of projectors to apply sequentially
        """
        self.projectors = projectors
        
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """
        Apply projectors sequentially.
        """
        result = params
        for projector in self.projectors:
            result = projector.project(result)
        return result
    
    def compression_ratio(self) -> float:
        """
        Total compression is product of individual ratios.
        """
        ratio = 1.0
        for projector in self.projectors:
            ratio *= projector.compression_ratio()
        return ratio


def compress_model(
    model: nn.Module,
    projector: CompressionProjector,
    inplace: bool = False
) -> nn.Module:
    """
    Compress all parameters of a model using a projector.
    
    Args:
        model: Neural network model
        projector: Compression projector
        inplace: Whether to modify model in-place
        
    Returns:
        Compressed model
    """
    if not inplace:
        model = type(model)()  # Create new instance
        model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in model.parameters():
            compressed = projector.project(param.data)
            param.data.copy_(compressed)
    
    return model


def measure_compression(
    original_model: nn.Module,
    compressed_model: nn.Module
) -> dict:
    """
    Measure compression statistics.
    
    Returns:
        Dictionary with compression metrics
    """
    # Count parameters
    orig_params = sum(p.numel() for p in original_model.parameters())
    comp_params = sum(p.numel() for p in compressed_model.parameters())
    
    # Estimate memory (assuming float32)
    orig_memory = orig_params * 4  # bytes
    
    # For compressed model, estimate based on unique values
    unique_values = set()
    for p in compressed_model.parameters():
        unique_values.update(p.flatten().tolist())
    
    bits_per_value = max(1, torch.tensor(len(unique_values)).log2().ceil().item())
    comp_memory = comp_params * bits_per_value / 8  # bytes
    
    # Compute distortion (L2 distance)
    distortion = 0.0
    for p_orig, p_comp in zip(original_model.parameters(), compressed_model.parameters()):
        distortion += torch.norm(p_orig - p_comp).item() ** 2
    distortion = distortion ** 0.5
    
    return {
        'original_params': orig_params,
        'compressed_params': comp_params,
        'original_memory_mb': orig_memory / (1024 ** 2),
        'compressed_memory_mb': comp_memory / (1024 ** 2),
        'compression_ratio': orig_memory / comp_memory,
        'distortion': distortion,
    }
