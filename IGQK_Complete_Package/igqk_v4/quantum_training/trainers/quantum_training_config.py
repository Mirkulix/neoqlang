"""
Quantum Training Configuration

This module defines the configuration for quantum-based neural network training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class QuantumTrainingConfig:
    """
    Configuration for Quantum LLM Training (v4.0).

    This unifies v2.0 vision with v4.0 advanced features.
    """

    # ==================== Model Architecture ====================
    model_type: Literal['GPT', 'BERT', 'T5', 'ViT', 'Whisper', 'MultiModal'] = 'GPT'
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: Optional[int] = None  # If None, defaults to 4 * d_model
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.1

    # ==================== Quantum Parameters ====================
    use_quantum: bool = True
    hbar: float = 0.1  # Quantum uncertainty (ℏ)
    gamma: float = 0.01  # Damping parameter (γ)
    use_fisher_metric: bool = True  # Use Fisher-Information metric for natural gradients
    quantum_ratio: float = 0.7  # Ratio of quantum vs classical updates (0.7 = 70% quantum)
    auto_switch: bool = True  # Automatically switch between QGF and Adam based on loss landscape

    # ==================== Compression Settings ====================
    train_compressed: bool = True  # NEW v4.0: Train directly in compressed space
    compression_method: Literal['ternary', 'binary', 'sparse', 'lowrank', 'auto'] = 'ternary'
    compression_target: float = 16.0  # Target compression ratio
    quality_target: float = 0.95  # Minimum quality retention (95%)

    # ==================== Advanced Math Frameworks (NEW v4.0) ====================
    use_hlwt: bool = True  # Hybrid Laplace-Wavelet Transform
    hlwt_wavelet_grid: tuple = (8, 8)  # Wavelet grid size (time, frequency)
    hlwt_wavelet_type: Literal['morlet', 'mexican_hat', 'haar'] = 'morlet'

    use_tlgt: bool = True  # Ternary Lie Group Theory
    tlgt_geodesic_steps: int = 5  # Number of geodesic steps
    tlgt_manifold_dim: Optional[int] = None  # Low-rank approximation dimension

    use_fchl: bool = False  # Fractional Calculus Hebbian Learning
    fchl_alpha: float = 0.7  # Fractional order (0 < α < 1)
    fchl_memory_length: int = 100  # Number of past steps to remember

    # ==================== Training Hyperparameters ====================
    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # ==================== Distributed Training (NEW v4.0) ====================
    distributed: bool = False
    strategy: Literal['ddp', 'fsdp', 'deepspeed'] = 'ddp'
    num_gpus: int = 1
    num_nodes: int = 1
    quantum_state_sharding: bool = False  # Shard quantum density matrix across GPUs

    # ==================== Auto-Tuning (NEW v4.0) ====================
    auto_tune: bool = False
    tuning_budget_hours: float = 24.0
    tuning_objective: Literal['loss', 'loss_and_size', 'throughput'] = 'loss_and_size'
    tuning_trials: int = 100

    # ==================== Hardware Acceleration (NEW v4.0) ====================
    use_custom_cuda: bool = True  # Use custom CUDA kernels for ternary ops
    use_fpga: bool = False  # Use FPGA accelerator (if available)
    use_tpu_t: bool = False  # Use TPU-T (Ternary Processing Unit)
    hardware_auto_detect: bool = True

    # ==================== Logging & Monitoring ====================
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    log_quantum_metrics: bool = True  # Log entropy, purity, etc.
    use_wandb: bool = False
    wandb_project: str = "igqk-v4"

    # ==================== Advanced Options ====================
    mixed_precision: bool = True  # FP16 training
    gradient_checkpointing: bool = False  # Memory-efficient training
    compile_model: bool = True  # torch.compile() for speedup

    # Multi-Modal specific (if model_type == 'MultiModal')
    multimodal_modalities: List[str] = field(default_factory=lambda: ['vision', 'language'])
    multimodal_fusion: Literal['quantum_entanglement', 'cross_attention', 'concat'] = 'quantum_entanglement'

    def __post_init__(self):
        """Validate and set defaults."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.train_compressed and not self.use_quantum:
            print("⚠️  Warning: train_compressed=True but use_quantum=False. Enabling quantum training.")
            self.use_quantum = True

        if self.distributed and self.num_gpus < 2:
            print("⚠️  Warning: distributed=True but num_gpus < 2. Setting distributed=False.")
            self.distributed = False

        if self.use_fchl and self.fchl_alpha <= 0 or self.fchl_alpha >= 1:
            raise ValueError(f"fchl_alpha must be in (0, 1), got {self.fchl_alpha}")

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)

    def summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("🔧 IGQK v4.0 - QUANTUM TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Model Type: {self.model_type}")
        print(f"Architecture: {self.n_layers} layers, {self.n_heads} heads, d={self.d_model}")
        print(f"\n🌟 Quantum Features:")
        print(f"  • Quantum Training: {self.use_quantum}")
        print(f"  • Quantum Ratio: {self.quantum_ratio*100:.0f}% (ℏ={self.hbar}, γ={self.gamma})")
        print(f"  • Fisher Metric: {self.use_fisher_metric}")
        print(f"  • Auto-Switch: {self.auto_switch}")
        print(f"\n🗜️  Compression:")
        print(f"  • Train Compressed: {self.train_compressed}")
        print(f"  • Method: {self.compression_method}")
        print(f"  • Target Ratio: {self.compression_target}×")
        print(f"\n🔬 Advanced Math:")
        print(f"  • HLWT (Laplace-Wavelet): {self.use_hlwt}")
        print(f"  • TLGT (Lie Groups): {self.use_tlgt}")
        print(f"  • FCHL (Fractional Calculus): {self.use_fchl}")
        if self.distributed:
            print(f"\n🌐 Distributed:")
            print(f"  • Strategy: {self.strategy}")
            print(f"  • GPUs: {self.num_gpus}, Nodes: {self.num_nodes}")
        if self.auto_tune:
            print(f"\n🤖 Auto-Tuning:")
            print(f"  • Budget: {self.tuning_budget_hours}h")
            print(f"  • Trials: {self.tuning_trials}")
        print(f"\n⚡ Hardware:")
        print(f"  • Custom CUDA: {self.use_custom_cuda}")
        print(f"  • FPGA: {self.use_fpga}")
        print(f"  • TPU-T: {self.use_tpu_t}")
        print("=" * 70)


# Preset configurations for common use cases
class ConfigPresets:
    """Preset configurations for common scenarios."""

    @staticmethod
    def small_gpt():
        """Small GPT model for testing."""
        return QuantumTrainingConfig(
            model_type='GPT',
            n_layers=6,
            n_heads=8,
            d_model=512,
            use_quantum=True,
            train_compressed=True,
        )

    @staticmethod
    def large_gpt():
        """Large GPT model (1B parameters)."""
        return QuantumTrainingConfig(
            model_type='GPT',
            n_layers=24,
            n_heads=16,
            d_model=1536,
            use_quantum=True,
            train_compressed=True,
            distributed=True,
            num_gpus=8,
        )

    @staticmethod
    def vision_transformer():
        """Vision Transformer configuration."""
        return QuantumTrainingConfig(
            model_type='ViT',
            n_layers=12,
            n_heads=12,
            d_model=768,
            use_quantum=True,
            train_compressed=True,
        )

    @staticmethod
    def multimodal_clip():
        """Multi-modal CLIP-like model."""
        return QuantumTrainingConfig(
            model_type='MultiModal',
            n_layers=12,
            d_model=512,
            use_quantum=True,
            train_compressed=True,
            multimodal_modalities=['vision', 'language'],
            multimodal_fusion='quantum_entanglement',
        )

    @staticmethod
    def auto_tuned():
        """Auto-tuned configuration."""
        return QuantumTrainingConfig(
            model_type='GPT',
            n_layers=12,
            d_model=768,
            auto_tune=True,
            tuning_budget_hours=12.0,
        )


if __name__ == "__main__":
    # Test configurations
    print("\n🧪 Testing Configuration Presets:\n")

    print("1. Small GPT:")
    config = ConfigPresets.small_gpt()
    config.summary()

    print("\n2. Multi-Modal CLIP:")
    config = ConfigPresets.multimodal_clip()
    config.summary()
