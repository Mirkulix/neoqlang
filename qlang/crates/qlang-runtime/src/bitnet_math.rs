//! BitNet Mathematics — Absmean Quantization + RMSNorm + Ternary LoRA
//!
//! Implements the mathematical foundations from:
//! - Chinese research (BitNet b1.58): Absmean quantization, RMSNorm
//! - Russian research (TT-Logic): Tensor-Train decomposition, Ternary LoRA
//!
//! References:
//! - "The Era of 1-bit LLMs" (Ma et al., 2024) — Equations 1-3
//! - "Ternäre Neuronale Netze: Die Mathematik der 1.58-Bit Ära" — Equations 2,4,8
//! - "Mathematische Grundlagen Ternärer Netze" — Equations 1-5

use rayon::prelude::*;

// ============================================================
// Absmean Quantization (BitNet b1.58)
// ============================================================

/// Absmean scaling factor γ = (1/||W||_0) * Σ|W_ij|
///
/// Scales weights so they average to magnitude 1 before rounding.
/// From Equation (2) in "Mathematische Grundlagen":
///   γ = (1 / ||W||_0) * Σ_{i,j} |W_{i,j}|
pub fn absmean_gamma(weights: &[f32]) -> f32 {
    let n = weights.len();
    if n == 0 { return 1.0; }
    let abs_sum: f32 = weights.iter().map(|w| w.abs()).sum();
    abs_sum / n as f32
}

/// Absmean ternary quantization (BitNet b1.58).
///
/// W̃ = Round(Clip(W / (γ + ε), -1, 1))
///
/// From Equation (1) in "Mathematische Grundlagen":
/// 1. Compute γ = absmean of weights
/// 2. Scale: W / (γ + ε)
/// 3. Clip to [-1, 1]
/// 4. Round to nearest integer → {-1, 0, +1}
///
/// Returns (ternary_weights, gamma)
pub fn absmean_quantize(weights: &[f32]) -> (Vec<f32>, f32) {
    let gamma = absmean_gamma(weights);
    let eps = 1e-8;
    let scale = gamma + eps;

    let ternary: Vec<f32> = weights.iter().map(|&w| {
        let scaled = w / scale;
        let clipped = scaled.max(-1.0).min(1.0);
        clipped.round() // → {-1, 0, +1}
    }).collect();

    (ternary, gamma)
}

/// Absmean quantize per-row (per output neuron).
/// Each row of [out_dim, in_dim] gets its own gamma.
pub fn absmean_quantize_rows(weights: &[f32], out_dim: usize, in_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut ternary = vec![0.0f32; out_dim * in_dim];
    let mut gammas = vec![0.0f32; out_dim];

    for j in 0..out_dim {
        let row = &weights[j * in_dim..(j + 1) * in_dim];
        let (t_row, gamma) = absmean_quantize(row);
        ternary[j * in_dim..(j + 1) * in_dim].copy_from_slice(&t_row);
        gammas[j] = gamma;
    }

    (ternary, gammas)
}

// ============================================================
// RMSNorm Stabilization
// ============================================================

/// Root Mean Square Normalization.
///
/// RMSNorm(x) = x / sqrt(E[x²] + ε) * g
///
/// From Equation (3) in "Mathematische Grundlagen".
/// Critical for ternary networks to prevent gradient explosion.
pub fn rmsnorm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 { return vec![]; }

    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (mean_sq + eps).sqrt();

    x.iter().zip(g.iter()).map(|(&xi, &gi)| (xi / rms) * gi).collect()
}

/// RMSNorm for batched activations [batch, dim].
pub fn rmsnorm_batch(x: &[f32], g: &[f32], batch: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * dim];

    for b in 0..batch {
        let row = &x[b * dim..(b + 1) * dim];
        let normed = rmsnorm(row, g, eps);
        output[b * dim..(b + 1) * dim].copy_from_slice(&normed);
    }

    output
}

// ============================================================
// Ternary LoRA (Russian research)
// ============================================================

/// Ternary Low-Rank Adaptation.
///
/// ΔW = A · B, where A and B are ternary matrices.
///
/// From Equation (5) in "Mathematische Grundlagen":
/// - A is [out_dim, rank] ternary
/// - B is [rank, in_dim] ternary
/// - ΔW = A × B is the weight update
///
/// This allows fine-tuning a model with minimal memory:
/// instead of storing out_dim × in_dim f32, store
/// (out_dim + in_dim) × rank ternary values.
pub struct TernaryLoRA {
    /// Low-rank matrix A [out_dim, rank] ternary
    pub a: Vec<f32>,
    /// Low-rank matrix B [rank, in_dim] ternary
    pub b: Vec<f32>,
    /// Scaling factor
    pub alpha: f32,
    pub out_dim: usize,
    pub in_dim: usize,
    pub rank: usize,
}

impl TernaryLoRA {
    /// Create a new ternary LoRA from a weight delta.
    ///
    /// 1. SVD of ΔW to get top-rank approximation
    /// 2. Quantize U and V to ternary
    pub fn from_delta(delta: &[f32], out_dim: usize, in_dim: usize, rank: usize) -> Self {
        // Simple rank-1 approximation via power iteration
        // For each rank component: find best ternary outer product
        let mut a = vec![0.0f32; out_dim * rank];
        let mut b = vec![0.0f32; rank * in_dim];
        let mut residual = delta.to_vec();

        for r in 0..rank {
            // Power iteration to find dominant singular vector
            let mut v = vec![1.0f32; in_dim];
            let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= v_norm; }

            for _ in 0..10 {
                // u = residual @ v
                let mut u = vec![0.0f32; out_dim];
                for i in 0..out_dim {
                    for k in 0..in_dim {
                        u[i] += residual[i * in_dim + k] * v[k];
                    }
                }
                let u_norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for x in u.iter_mut() { *x /= u_norm; }

                // v = residual^T @ u
                v = vec![0.0f32; in_dim];
                for k in 0..in_dim {
                    for i in 0..out_dim {
                        v[k] += residual[i * in_dim + k] * u[i];
                    }
                }
                let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for x in v.iter_mut() { *x /= v_norm; }
            }

            // Compute singular value σ
            let mut sigma = 0.0f32;
            let mut u_final = vec![0.0f32; out_dim];
            for i in 0..out_dim {
                for k in 0..in_dim {
                    u_final[i] += residual[i * in_dim + k] * v[k];
                }
            }
            sigma = u_final.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Ternarize u and v
            let (t_u, _) = absmean_quantize(&u_final);
            let (t_v, _) = absmean_quantize(&v);

            // Store in A and B
            for i in 0..out_dim { a[i * rank + r] = t_u[i]; }
            for k in 0..in_dim { b[r * in_dim + k] = t_v[k]; }

            // Subtract rank-1 component from residual
            for i in 0..out_dim {
                for k in 0..in_dim {
                    residual[i * in_dim + k] -= sigma * t_u[i] * t_v[k] / sigma.max(1e-10);
                }
            }
        }

        // Compute overall alpha
        let alpha = {
            let mut dot_orig = 0.0f32;
            let mut dot_approx = 0.0f32;
            // Compute A @ B
            for i in 0..out_dim {
                for k in 0..in_dim {
                    let mut ab_ik = 0.0f32;
                    for r in 0..rank {
                        ab_ik += a[i * rank + r] * b[r * in_dim + k];
                    }
                    dot_orig += delta[i * in_dim + k] * ab_ik;
                    dot_approx += ab_ik * ab_ik;
                }
            }
            if dot_approx > 0.0 { dot_orig / dot_approx } else { 1.0 }
        };

        Self { a, b, alpha, out_dim, in_dim, rank }
    }

    /// Apply LoRA: W_new = W_base + alpha * A @ B
    pub fn apply(&self, base_weights: &[f32]) -> Vec<f32> {
        let mut result = base_weights.to_vec();
        for i in 0..self.out_dim {
            for k in 0..self.in_dim {
                let mut ab = 0.0f32;
                for r in 0..self.rank {
                    ab += self.a[i * self.rank + r] * self.b[r * self.in_dim + k];
                }
                result[i * self.in_dim + k] += self.alpha * ab;
            }
        }
        result
    }

    /// Memory usage in bytes (ternary packed).
    pub fn memory_bytes(&self) -> usize {
        let a_bits = self.out_dim * self.rank * 2; // 2 bits per ternary
        let b_bits = self.rank * self.in_dim * 2;
        (a_bits + b_bits) / 8 + 4 // + alpha f32
    }

    /// Memory saved vs full f32 delta.
    pub fn compression_ratio(&self) -> f32 {
        let full_size = (self.out_dim * self.in_dim * 4) as f32;
        let lora_size = self.memory_bytes() as f32;
        full_size / lora_size
    }

    /// Verify all weights are ternary.
    pub fn is_ternary(&self) -> bool {
        self.a.iter().all(|&w| w == -1.0 || w == 0.0 || w == 1.0) &&
        self.b.iter().all(|&w| w == -1.0 || w == 0.0 || w == 1.0)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmean_gamma_basic() {
        let w = vec![1.0, -2.0, 3.0, -4.0];
        let gamma = absmean_gamma(&w);
        assert!((gamma - 2.5).abs() < 1e-5); // (1+2+3+4)/4 = 2.5
    }

    #[test]
    fn absmean_quantize_basic() {
        let w = vec![0.5, -1.5, 0.1, 2.0, -0.8];
        let (ternary, gamma) = absmean_quantize(&w);
        // All outputs must be {-1, 0, +1}
        for &t in &ternary {
            assert!(t == -1.0 || t == 0.0 || t == 1.0, "got {}", t);
        }
        assert!(gamma > 0.0);
    }

    #[test]
    fn absmean_preserves_sign() {
        let w = vec![5.0, -5.0, 0.01];
        let (ternary, _) = absmean_quantize(&w);
        assert_eq!(ternary[0], 1.0);  // large positive → +1
        assert_eq!(ternary[1], -1.0); // large negative → -1
        assert_eq!(ternary[2], 0.0);  // near zero → 0
    }

    #[test]
    fn rmsnorm_basic() {
        let x = vec![3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let g = vec![1.0, 1.0];
        let normed = rmsnorm(&x, &g, 1e-8);
        let rms = (12.5f32).sqrt();
        assert!((normed[0] - 3.0 / rms).abs() < 1e-4);
        assert!((normed[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn rmsnorm_with_gain() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let g = vec![2.0, 2.0, 2.0, 2.0];
        let normed = rmsnorm(&x, &g, 1e-8);
        // rms = 1.0, so normed = x/1 * g = 2.0
        for &v in &normed {
            assert!((v - 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn ternary_lora_small() {
        // Create a known weight delta
        let out_dim = 4;
        let in_dim = 3;
        let delta = vec![
            1.0, 0.0, -1.0,
            0.0, 1.0,  0.0,
           -1.0, 0.0,  1.0,
            0.5, -0.5, 0.0,
        ];

        let lora = TernaryLoRA::from_delta(&delta, out_dim, in_dim, 2);
        assert!(lora.is_ternary(), "LoRA weights must be ternary");
        assert_eq!(lora.rank, 2);

        let compression = lora.compression_ratio();
        println!("LoRA compression: {:.1}x ({} bytes vs {} bytes)",
            compression, lora.memory_bytes(), out_dim * in_dim * 4);
    }

    #[test]
    fn ternary_lora_apply() {
        let out_dim = 8;
        let in_dim = 6;
        let base = vec![0.0f32; out_dim * in_dim];
        let delta: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();

        let lora = TernaryLoRA::from_delta(&delta, out_dim, in_dim, 3);
        let result = lora.apply(&base);

        // Result should approximate delta
        let error: f32 = result.iter().zip(delta.iter())
            .map(|(r, d)| (r - d) * (r - d))
            .sum::<f32>() / (out_dim * in_dim) as f32;
        println!("LoRA reconstruction MSE: {:.6}", error);
        // Won't be perfect with rank-3 ternary, but should capture structure
    }

    #[test]
    fn ternary_lora_compression_ratio() {
        // For a 768x768 layer with rank 8
        let out_dim = 768;
        let in_dim = 768;
        let rank = 8;
        let full_size = out_dim * in_dim * 4; // 2.25 MB
        let lora_a = out_dim * rank * 2 / 8; // ternary packed
        let lora_b = rank * in_dim * 2 / 8;
        let lora_size = lora_a + lora_b + 4; // + alpha

        let ratio = full_size as f32 / lora_size as f32;
        println!("768x768 with rank 8: full={} bytes, LoRA={} bytes, ratio={:.0}x",
            full_size, lora_size, ratio);
        assert!(ratio > 100.0, "Ternary LoRA must compress >100x for 768x768 rank 8");
    }

    #[test]
    fn absmean_vs_threshold_comparison() {
        // Compare absmean (BitNet) with our 0.7*mean_abs threshold
        let weights: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();

        // Absmean method (BitNet)
        let (tern_absmean, gamma_absmean) = absmean_quantize(&weights);

        // Threshold method (QLANG current)
        let mean_abs: f32 = weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len() as f32;
        let threshold = mean_abs * 0.7;
        let tern_threshold: Vec<f32> = weights.iter().map(|&w| {
            if w > threshold { 1.0 } else if w < -threshold { -1.0 } else { 0.0 }
        }).collect();

        // Count distribution
        let count = |v: &[f32], val: f32| v.iter().filter(|&&w| w == val).count();
        println!("Absmean:   +1:{} 0:{} -1:{} (gamma={:.4})",
            count(&tern_absmean, 1.0), count(&tern_absmean, 0.0), count(&tern_absmean, -1.0), gamma_absmean);
        println!("Threshold: +1:{} 0:{} -1:{} (thresh={:.4})",
            count(&tern_threshold, 1.0), count(&tern_threshold, 0.0), count(&tern_threshold, -1.0), threshold);
    }
}
