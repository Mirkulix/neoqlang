"""
Statistical manifolds with Fisher information metric.

Implements the geometry of parameter spaces for neural networks.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable
from abc import ABC, abstractmethod


class StatisticalManifold(ABC):
    """
    Abstract base class for statistical manifolds.
    
    A statistical manifold is equipped with the Fisher information metric,
    which measures the distinguishability of nearby distributions.
    """
    
    @abstractmethod
    def fisher_metric(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fisher information matrix at current parameters.
        
        Args:
            model: Neural network model
            data: Input data batch
            target: Target labels/values
            
        Returns:
            Fisher information matrix (n_params, n_params)
        """
        pass
    
    @abstractmethod
    def geodesic_distance(
        self,
        params1: torch.Tensor,
        params2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geodesic distance between two parameter configurations.
        
        Args:
            params1: First parameter vector
            params2: Second parameter vector
            
        Returns:
            Scalar distance
        """
        pass
    
    def natural_gradient(
        self,
        gradient: torch.Tensor,
        fisher: torch.Tensor,
        damping: float = 1e-4
    ) -> torch.Tensor:
        """
        Compute natural gradient: G⁻¹ ∇L
        
        Args:
            gradient: Standard gradient
            fisher: Fisher information matrix
            damping: Damping factor for numerical stability
            
        Returns:
            Natural gradient
        """
        # Add damping for numerical stability
        fisher_damped = fisher + damping * torch.eye(fisher.shape[0], device=fisher.device)
        
        # Solve: G * nat_grad = grad
        nat_grad = torch.linalg.solve(fisher_damped, gradient)
        
        return nat_grad


class EmpiricalFisherManifold(StatisticalManifold):
    """
    Empirical Fisher information manifold.
    
    Computes Fisher metric using empirical gradients:
    G = E[∇log p(y|x) ∇log p(y|x)ᵀ]
    """
    
    def __init__(self, n_samples: int = 100):
        """
        Args:
            n_samples: Number of samples for empirical estimation
        """
        self.n_samples = n_samples
        
    def fisher_metric(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute empirical Fisher information matrix.
        """
        model.eval()
        
        # Get all parameters as a single vector
        params = torch.cat([p.flatten() for p in model.parameters()])
        n_params = params.shape[0]
        
        # Initialize Fisher matrix
        fisher = torch.zeros(n_params, n_params, device=params.device)
        
        # Sample subset of data
        n_data = min(self.n_samples, data.shape[0])
        indices = torch.randperm(data.shape[0])[:n_data]
        
        # Compute gradients for each sample
        for idx in indices:
            x = data[idx:idx+1]
            y = target[idx:idx+1]
            
            # Forward pass
            output = model(x)
            
            # Compute log probability (assume cross-entropy)
            log_prob = -nn.functional.cross_entropy(output, y, reduction='sum')
            
            # Compute gradient
            model.zero_grad()
            log_prob.backward()
            
            # Get gradient vector
            grad = torch.cat([p.grad.flatten() for p in model.parameters()])
            
            # Accumulate outer product
            fisher += torch.outer(grad, grad)
        
        # Average
        fisher = fisher / n_data
        
        model.train()
        return fisher
    
    def geodesic_distance(
        self,
        params1: torch.Tensor,
        params2: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate geodesic distance using Euclidean distance.
        
        For a more accurate computation, one would need to integrate
        along the geodesic, which is computationally expensive.
        """
        return torch.norm(params1 - params2)


class DiagonalFisherManifold(StatisticalManifold):
    """
    Diagonal approximation of Fisher information manifold.
    
    Assumes Fisher matrix is diagonal, which is much more efficient
    to compute and invert.
    """
    
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
        
    def fisher_metric(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diagonal Fisher information matrix.
        """
        model.eval()
        
        # Get all parameters
        params = torch.cat([p.flatten() for p in model.parameters()])
        n_params = params.shape[0]
        
        # Initialize diagonal Fisher
        fisher_diag = torch.zeros(n_params, device=params.device)
        
        # Sample subset
        n_data = min(self.n_samples, data.shape[0])
        indices = torch.randperm(data.shape[0])[:n_data]
        
        for idx in indices:
            x = data[idx:idx+1]
            y = target[idx:idx+1]
            
            output = model(x)
            log_prob = -nn.functional.cross_entropy(output, y, reduction='sum')
            
            model.zero_grad()
            log_prob.backward()
            
            grad = torch.cat([p.grad.flatten() for p in model.parameters()])
            
            # Accumulate squared gradients
            fisher_diag += grad ** 2
        
        fisher_diag = fisher_diag / n_data
        
        # Return as diagonal matrix
        fisher = torch.diag(fisher_diag)
        
        model.train()
        return fisher
    
    def natural_gradient(
        self,
        gradient: torch.Tensor,
        fisher: torch.Tensor,
        damping: float = 1e-4
    ) -> torch.Tensor:
        """
        Efficient natural gradient for diagonal Fisher.
        """
        # Extract diagonal
        fisher_diag = torch.diag(fisher)
        
        # Element-wise division
        nat_grad = gradient / (fisher_diag + damping)
        
        return nat_grad
    
    def geodesic_distance(
        self,
        params1: torch.Tensor,
        params2: torch.Tensor
    ) -> torch.Tensor:
        """Euclidean approximation."""
        return torch.norm(params1 - params2)


class BlockDiagonalFisherManifold(StatisticalManifold):
    """
    Block-diagonal approximation of Fisher information.
    
    Treats each layer's parameters as a separate block, which is
    more accurate than diagonal but still efficient.
    """
    
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
        
    def fisher_metric(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute block-diagonal Fisher information matrix.
        """
        model.eval()
        
        # Compute Fisher for each layer separately
        layer_fishers = []
        
        n_data = min(self.n_samples, data.shape[0])
        indices = torch.randperm(data.shape[0])[:n_data]
        
        for param in model.parameters():
            n_param = param.numel()
            fisher_block = torch.zeros(n_param, n_param, device=param.device)
            
            for idx in indices:
                x = data[idx:idx+1]
                y = target[idx:idx+1]
                
                output = model(x)
                log_prob = -nn.functional.cross_entropy(output, y, reduction='sum')
                
                model.zero_grad()
                log_prob.backward()
                
                if param.grad is not None:
                    grad = param.grad.flatten()
                    fisher_block += torch.outer(grad, grad)
            
            fisher_block = fisher_block / n_data
            layer_fishers.append(fisher_block)
        
        # Construct block-diagonal matrix
        total_params = sum(f.shape[0] for f in layer_fishers)
        fisher = torch.zeros(total_params, total_params, device=data.device)
        
        offset = 0
        for block in layer_fishers:
            size = block.shape[0]
            fisher[offset:offset+size, offset:offset+size] = block
            offset += size
        
        model.train()
        return fisher
    
    def geodesic_distance(
        self,
        params1: torch.Tensor,
        params2: torch.Tensor
    ) -> torch.Tensor:
        """Euclidean approximation."""
        return torch.norm(params1 - params2)
