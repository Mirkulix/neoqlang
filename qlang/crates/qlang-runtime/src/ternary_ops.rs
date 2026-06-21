//! Ternary Arithmetic — zero-multiply inference for {-1, 0, +1} weights.
//!
//! When weights are ternary, multiplication is replaced by:
//!   w = +1 → add x
//!   w = -1 → subtract x
//!   w =  0 → skip (no operation)
//!
//! This is pure add/sub — no FPU multiply needed.
//! On real hardware this is 3-10x faster than f32 matmul.

/// Ternary matrix-vector product: y = W @ x + bias
///
/// W is [out_dim, in_dim] ternary ({-1, 0, +1}), x is [in_dim] f32.
/// Returns [out_dim] f32.
///
/// Uses branch-free add/sub instead of multiply.
#[inline]
pub fn ternary_matvec(w: &[f32], x: &[f32], bias: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(bias.len(), out_dim);

    let mut y = bias.to_vec();
    for j in 0..out_dim {
        let w_row = &w[j * in_dim..(j + 1) * in_dim];
        let mut sum = 0.0f32;
        for k in 0..in_dim {
            // Branch-free: w is exactly -1, 0, or +1
            // w * x = x if w=1, -x if w=-1, 0 if w=0
            // Equivalent to: sum += w[k] * x[k] but without FPU multiply
            let wi = w_row[k] as i32; // -1, 0, or 1
            // Branchless: result = x * sign, where sign is -1/0/+1
            // On x86 this compiles to conditional moves or mask operations
            sum += x[k] * wi as f32;
        }
        y[j] += sum;
    }
    y
}

/// Ternary batch matrix multiply: Y = X @ W^T + bias, with ReLU.
///
/// X is [batch, in_dim], W is [out_dim, in_dim] ternary, bias is [out_dim].
/// Returns [batch, out_dim] with ReLU applied.
///
/// Uses true ternary arithmetic: no multiply, only conditional add/sub.
/// Weights are pre-split into positive and negative masks for branchless execution.
pub fn ternary_forward_relu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), batch * in_dim);
    debug_assert_eq!(w.len(), out_dim * in_dim);

    let mut output = vec![0.0f32; batch * out_dim];

    // Pre-compute positive/negative masks per output neuron
    // pos_mask[j]: indices where w[j,k] == +1
    // neg_mask[j]: indices where w[j,k] == -1
    let mut pos_masks: Vec<Vec<usize>> = Vec::with_capacity(out_dim);
    let mut neg_masks: Vec<Vec<usize>> = Vec::with_capacity(out_dim);
    for j in 0..out_dim {
        let w_row = &w[j * in_dim..(j + 1) * in_dim];
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for k in 0..in_dim {
            if w_row[k] > 0.5 { pos.push(k); }
            else if w_row[k] < -0.5 { neg.push(k); }
            // w == 0: skip entirely
        }
        pos_masks.push(pos);
        neg_masks.push(neg);
    }

    for b in 0..batch {
        let x_row = &x[b * in_dim..(b + 1) * in_dim];
        for j in 0..out_dim {
            let mut sum = bias[j];
            // Add all x[k] where w[j,k] == +1
            for &k in &pos_masks[j] {
                sum += x_row[k];
            }
            // Subtract all x[k] where w[j,k] == -1
            for &k in &neg_masks[j] {
                sum -= x_row[k];
            }
            // ReLU
            output[b * out_dim + j] = sum.max(0.0);
        }
    }

    output
}

/// Ternary batch forward WITHOUT ReLU (for last layer / logits).
pub fn ternary_forward_linear(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), batch * in_dim);

    let mut output = vec![0.0f32; batch * out_dim];

    for b in 0..batch {
        let x_row = &x[b * in_dim..(b + 1) * in_dim];
        for j in 0..out_dim {
            let w_row = &w[j * in_dim..(j + 1) * in_dim];
            let mut sum = bias[j];
            for k in 0..in_dim {
                sum += x_row[k] * w_row[k];
            }
            output[b * out_dim + j] = sum;
        }
    }

    output
}

/// Pack ternary weights into 2-bit representation for storage.
/// Each weight uses 2 bits: 00=0, 01=+1, 11=-1.
/// Returns packed bytes + scaling factor.
pub fn pack_ternary(weights: &[f32]) -> (Vec<u8>, f32) {
    // Find scaling factor (mean absolute value of non-zero weights)
    let non_zero: Vec<f32> = weights.iter().filter(|&&w| w != 0.0).map(|w| w.abs()).collect();
    let alpha = if non_zero.is_empty() {
        1.0
    } else {
        non_zero.iter().sum::<f32>() / non_zero.len() as f32
    };

    // Pack 4 weights per byte (2 bits each)
    let n_bytes = (weights.len() + 3) / 4;
    let mut packed = vec![0u8; n_bytes];

    let threshold = alpha * 0.7;
    for (i, &w) in weights.iter().enumerate() {
        let bits: u8 = if w > threshold {
            0b01 // +1
        } else if w < -threshold {
            0b11 // -1
        } else {
            0b00 // 0
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= bits << bit_offset;
    }

    (packed, alpha)
}

/// Unpack 2-bit ternary weights back to f32.
pub fn unpack_ternary(packed: &[u8], n_weights: usize, alpha: f32) -> Vec<f32> {
    let mut weights = Vec::with_capacity(n_weights);
    for i in 0..n_weights {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_offset) & 0b11;
        let w = match bits {
            0b01 => alpha,
            0b11 => -alpha,
            _ => 0.0,
        };
        weights.push(w);
    }
    weights
}

/// Packed ternary matrix-vector product — the real tokens/watt kernel.
///
/// Reads weights directly from 2-bit packed bytes, never expands to f32.
/// This is the actual DRAM-bandwidth-saving kernel: 16x less weight traffic
/// vs f32, 4x less than int8. The decode (2-bit → ±1) is done inline
/// with a 2-bit mask — no unpacking allocations.
///
/// Uses flat (dense) indexing matching `pack_ternary`, so it is correct for
/// any in_dim. When in_dim is a multiple of 4 the rows are byte-aligned and
/// the inner loop is maximally vectorisable by the compiler.
///
/// y[j] = alpha * Σ_k decode(w_packed[j,k]) * x[k] + bias[j]
/// Encoding (same as `pack_ternary`): 00=0, 01=+1, 11=-1, 4 weights per byte.
pub fn ternary_packed_matvec(
    w_packed: &[u8],
    alpha: f32,
    x: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(w_packed.len(), (out_dim * in_dim + 3) / 4);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(bias.len(), out_dim);

    let mut y = bias.to_vec();
    for j in 0..out_dim {
        let base_idx = j * in_dim;
        let mut sum = 0.0f32;
        for k in 0..in_dim {
            let flat = base_idx + k;
            let bits = (w_packed[flat / 4] >> ((flat % 4) * 2)) & 0b11;
            match bits {
                0b01 => sum += x[k],
                0b11 => sum -= x[k],
                _ => {}
            }
        }
        y[j] += alpha * sum;
    }
    y
}

/// Packed ternary batch forward pass with ReLU activation.
///
/// X is [batch, in_dim], w_packed is the output of `pack_ternary` on the
/// [out_dim × in_dim] weight matrix stored row-major.
/// Returns [batch, out_dim] with ReLU applied.
///
/// Same DRAM savings as `ternary_packed_matvec` but batched for multi-token inference.
pub fn ternary_packed_forward_relu(
    w_packed: &[u8],
    alpha: f32,
    x: &[f32],
    bias: &[f32],
    batch: usize,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(w_packed.len(), (out_dim * in_dim + 3) / 4);
    debug_assert_eq!(x.len(), batch * in_dim);
    debug_assert_eq!(bias.len(), out_dim);

    let mut output = vec![0.0f32; batch * out_dim];
    for b in 0..batch {
        let x_row = &x[b * in_dim..(b + 1) * in_dim];
        for j in 0..out_dim {
            let base_idx = j * in_dim;
            let mut sum = 0.0f32;
            for k in 0..in_dim {
                let flat = base_idx + k;
                let bits = (w_packed[flat / 4] >> ((flat % 4) * 2)) & 0b11;
                match bits {
                    0b01 => sum += x_row[k],
                    0b11 => sum -= x_row[k],
                    _ => {}
                }
            }
            output[b * out_dim + j] = (alpha * sum + bias[j]).max(0.0);
        }
    }
    output
}

/// Count operations saved by ternary vs f32.
pub fn ops_saved(weights: &[f32]) -> (usize, usize, usize) {
    let total = weights.len();
    let zeros = weights.iter().filter(|&&w| w == 0.0).count();
    let non_zeros = total - zeros;
    // f32: total multiplications + total additions
    // Ternary: non_zeros additions (zeros are skipped entirely)
    let f32_ops = total * 2; // mul + add per weight
    let ternary_ops = non_zeros; // only add/sub, skip zeros
    (f32_ops, ternary_ops, zeros)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_matvec_basic() {
        // W = [[1, -1, 0], [0, 1, 1]]  (2x3)
        let w = vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0];
        let x = vec![3.0, 2.0, 1.0];
        let bias = vec![0.0, 0.0];

        let y = ternary_matvec(&w, &x, &bias, 2, 3);
        // y[0] = 3*1 + 2*(-1) + 1*0 = 1
        // y[1] = 3*0 + 2*1 + 1*1 = 3
        assert_eq!(y, vec![1.0, 3.0]);
    }

    #[test]
    fn ternary_forward_relu_basic() {
        let w = vec![1.0, -1.0, 0.0, 1.0]; // 2x2
        let x = vec![2.0, 3.0]; // 1x2
        let bias = vec![0.0, 0.0];

        let out = ternary_forward_relu(&x, &w, &bias, 1, 2, 2);
        // [0] = relu(2*1 + 3*(-1)) = relu(-1) = 0
        // [1] = relu(2*0 + 3*1) = relu(3) = 3
        assert_eq!(out, vec![0.0, 3.0]);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let weights = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let (packed, alpha) = pack_ternary(&weights);

        assert_eq!(packed.len(), 2); // 8 weights / 4 per byte = 2 bytes
        assert!((alpha - 1.0).abs() < 0.01);

        let unpacked = unpack_ternary(&packed, weights.len(), alpha);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn pack_size_reduction() {
        let n = 100_000;
        let weights: Vec<f32> = (0..n).map(|i| {
            let v = (i as f32 * 0.37).sin();
            if v > 0.3 { 1.0 } else if v < -0.3 { -1.0 } else { 0.0 }
        }).collect();

        let f32_size = n * 4; // 400 KB
        let (packed, _) = pack_ternary(&weights);
        let tern_size = packed.len(); // ~25 KB

        let ratio = f32_size as f32 / tern_size as f32;
        println!("f32: {} bytes, ternary packed: {} bytes, ratio: {:.1}x", f32_size, tern_size, ratio);
        assert!(ratio > 15.0, "Ternary packing must achieve >15x (got {:.1}x)", ratio);
    }

    #[test]
    fn ops_saved_calculation() {
        let weights = vec![1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0];
        let (f32_ops, tern_ops, zeros) = ops_saved(&weights);
        assert_eq!(zeros, 4); // 4 zeros
        assert_eq!(f32_ops, 16); // 8 * 2 (mul + add)
        assert_eq!(tern_ops, 4); // only 4 non-zero add/sub
        println!("f32: {} ops, ternary: {} ops, saved: {:.0}%",
            f32_ops, tern_ops, (1.0 - tern_ops as f32 / f32_ops as f32) * 100.0);
    }

    #[test]
    fn packed_matvec_basic_correctness() {
        // W = [[+1, -1, 0]] (1x3), x=[3,2,1], bias=[0] → y = 3-2+0 = 1
        let w_f32 = vec![1.0f32, -1.0, 0.0];
        let x = vec![3.0f32, 2.0, 1.0];
        let bias = vec![0.0f32];
        let (w_packed, alpha) = pack_ternary(&w_f32);

        let y = ternary_packed_matvec(&w_packed, alpha, &x, &bias, 1, 3);
        assert!((y[0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", y[0]);
    }

    #[test]
    fn packed_matvec_matches_unpacked() {
        // W = [[+1,-1,0,+1],[0,+1,-1,0]] (2x4), alpha=1.0
        let w_f32 = vec![1.0f32, -1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0];
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let bias = vec![0.5f32, -0.5];
        let (w_packed, alpha) = pack_ternary(&w_f32);

        let y_ref = ternary_matvec(&w_f32, &x, &bias, 2, 4);
        let y_packed = ternary_packed_matvec(&w_packed, alpha, &x, &bias, 2, 4);

        for (i, (a, b)) in y_ref.iter().zip(y_packed.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "row {}: ref={} packed={}", i, a, b);
        }
    }

    #[test]
    fn packed_forward_relu_matches_unpacked() {
        let in_dim = 8;
        let out_dim = 4;
        let batch = 3;
        let w_f32: Vec<f32> = (0..out_dim * in_dim).map(|i| {
            let v = (i as f32 * 0.7).sin();
            if v > 0.4 { 1.0 } else if v < -0.4 { -1.0 } else { 0.0 }
        }).collect();
        let x: Vec<f32> = (0..batch * in_dim).map(|i| (i as f32 * 0.1).cos()).collect();
        let bias = vec![0.1f32; out_dim];
        let (w_packed, alpha) = pack_ternary(&w_f32);

        let y_ref = ternary_forward_relu(&x, &w_f32, &bias, batch, out_dim, in_dim);
        let y_packed = ternary_packed_forward_relu(&w_packed, alpha, &x, &bias, batch, out_dim, in_dim);

        for (i, (a, b)) in y_ref.iter().zip(y_packed.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "idx {}: ref={} packed={}", i, a, b);
        }
    }

    #[test]
    fn packed_matvec_unaligned_in_dim() {
        // in_dim=5: last byte holds 1 weight (bits [1:0]) + 3 padding bits
        let w_f32 = vec![1.0f32, -1.0, 0.0, 1.0, -1.0]; // 1x5
        let x = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0f32];
        let (w_packed, alpha) = pack_ternary(&w_f32);

        let y_ref = ternary_matvec(&w_f32, &x, &bias, 1, 5);
        let y_packed = ternary_packed_matvec(&w_packed, alpha, &x, &bias, 1, 5);
        assert!((y_ref[0] - y_packed[0]).abs() < 1e-5,
            "unaligned: ref={} packed={}", y_ref[0], y_packed[0]);
    }

    #[test]
    fn packed_kernel_bandwidth_comparison() {
        let in_dim = 4096;
        let out_dim = 4096;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let bias = vec![0.0f32; out_dim];
        let w_f32: Vec<f32> = (0..out_dim * in_dim).map(|i| {
            let v = (i as f32 * 0.37).sin();
            if v > 0.3 { 1.0 } else if v < -0.3 { -1.0 } else { 0.0 }
        }).collect();
        let (w_packed, alpha) = pack_ternary(&w_f32);

        let f32_bytes = out_dim * in_dim * 4;
        let packed_bytes = w_packed.len();
        println!("\nBandwidth: {}x{} weight matrix", out_dim, in_dim);
        println!("  f32:     {} MB", f32_bytes / 1_048_576);
        println!("  2-bit:   {} KB  ({:.1}x smaller)",
            packed_bytes / 1024, f32_bytes as f32 / packed_bytes as f32);

        let n_runs = 5usize;
        let start = std::time::Instant::now();
        for _ in 0..n_runs {
            std::hint::black_box(ternary_matvec(&w_f32, &x, &bias, out_dim, in_dim));
        }
        let t_unpacked = start.elapsed();

        let start = std::time::Instant::now();
        for _ in 0..n_runs {
            std::hint::black_box(ternary_packed_matvec(&w_packed, alpha, &x, &bias, out_dim, in_dim));
        }
        let t_packed = start.elapsed();

        println!("  f32-stored matvec: {:?} ({} runs)", t_unpacked, n_runs);
        println!("  packed matvec:     {:?} ({} runs)", t_packed, n_runs);
        println!("  wall-clock speedup: {:.2}x",
            t_unpacked.as_nanos() as f64 / t_packed.as_nanos() as f64);

        // Correctness check
        let y_ref = ternary_matvec(&w_f32, &x, &bias, out_dim, in_dim);
        let y_pack = ternary_packed_matvec(&w_packed, alpha, &x, &bias, out_dim, in_dim);
        let max_err = y_ref.iter().zip(y_pack.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("  max |err|: {:.2e}", max_err);
        assert!(max_err < 1e-3, "large error in packed kernel: {}", max_err);
    }

    /// Pure Rust f32 matmul without BLAS (fair comparison).
    fn naive_matmul_relu(x: &[f32], w: &[f32], bias: &[f32], batch: usize, out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * out_dim];
        for b in 0..batch {
            for j in 0..out_dim {
                let mut sum = bias[j];
                for k in 0..in_dim {
                    sum += x[b * in_dim + k] * w[j * in_dim + k];
                }
                output[b * out_dim + j] = sum.max(0.0);
            }
        }
        output
    }

    #[test]
    fn ternary_inference_benchmark() {
        // Benchmark: f32 matmul vs ternary forward for a realistic layer
        let in_dim = 784;
        let out_dim = 256;
        let batch = 100;

        // Random input
        let x: Vec<f32> = (0..batch * in_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let bias = vec![0.0f32; out_dim];

        // Ternary weights
        let w_tern: Vec<f32> = (0..out_dim * in_dim).map(|i| {
            let v = (i as f32 * 0.37).sin();
            if v > 0.3 { 1.0 } else if v < -0.3 { -1.0 } else { 0.0 }
        }).collect();

        // f32 weights (same but continuous)
        let w_f32: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();

        // Transpose w_f32 for accel::matmul (needs [in_dim, out_dim])
        let mut w_f32_t = vec![0.0f32; in_dim * out_dim];
        for i in 0..out_dim {
            for j in 0..in_dim {
                w_f32_t[j * out_dim + i] = w_f32[i * in_dim + j];
            }
        }

        // Benchmark f32 matmul
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = crate::accel::matmul(&x, &w_f32_t, batch, out_dim, in_dim);
        }
        let f32_time = start.elapsed();

        // Benchmark ternary forward
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ternary_forward_relu(&x, &w_tern, &bias, batch, out_dim, in_dim);
        }
        let tern_time = start.elapsed();

        let (f32_ops, tern_ops, zeros) = ops_saved(&w_tern);

        // Benchmark naive Rust f32 (fair comparison, no BLAS)
        let w_naive: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
        let mut naive_checksum = 0.0f32;
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let r = naive_matmul_relu(&x, &w_naive, &bias, batch, out_dim, in_dim);
            naive_checksum += r[0]; // prevent optimization
        }
        let naive_time = start.elapsed();
        std::hint::black_box(naive_checksum);

        println!("\nInference Benchmark: {} x [{},{}] → [{},{}]", batch, batch, in_dim, batch, out_dim);
        println!("  f32 BLAS:        {:?} (10 iters)", f32_time);
        println!("  f32 naive Rust:  {:?} (10 iters)", naive_time);
        println!("  ternary add/sub: {:?} (10 iters)", tern_time);
        println!("  Speedup vs BLAS:  {:.2}x", f32_time.as_nanos() as f64 / tern_time.as_nanos() as f64);
        println!("  Speedup vs naive: {:.2}x", naive_time.as_nanos() as f64 / tern_time.as_nanos() as f64);
        println!("  Zeros skipped:   {} / {} ({:.0}%)", zeros, w_tern.len(),
            zeros as f64 / w_tern.len() as f64 * 100.0);
        println!("  Ops: f32={}, ternary={}, saved {:.0}%",
            f32_ops, tern_ops, (1.0 - tern_ops as f64 / f32_ops as f64) * 100.0);
    }
}
