//! Transformer operations for QLANG.
//!
//! Implements the core building blocks of Transformer architectures:
//! - Scaled dot-product attention
//! - Multi-head attention
//! - Layer normalization
//! - GELU activation
//! - Positional encoding
//!
//! These run on the QLANG runtime (Phase 1: Rust, Phase 2: LLVM JIT).

use crate::autograd::Tape;

/// Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
///
/// Input: [batch, seq_len, d_model]  (flattened as [batch * seq_len, d_model])
/// Output: same shape
pub fn layer_norm(tape: &mut Tape, input: usize, d_model: usize, eps: f32) -> usize {
    let data = tape.value(input).to_vec();
    let n = data.len();
    let n_vectors = n / d_model;

    let mut normed = vec![0.0f32; n];

    for v in 0..n_vectors {
        let offset = v * d_model;
        let slice = &data[offset..offset + d_model];

        // Compute mean
        let mean: f32 = slice.iter().sum::<f32>() / d_model as f32;

        // Compute variance
        let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / d_model as f32;

        // Normalize
        let std = (var + eps).sqrt();
        for j in 0..d_model {
            normed[offset + j] = (slice[j] - mean) / std;
        }
    }

    let shape = tape.values[input].shape.clone();
    tape.variable(normed, shape)
}

/// GELU activation: x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
pub fn gelu(tape: &mut Tape, input: usize) -> usize {
    let data = tape.value(input).to_vec();
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();

    let result: Vec<f32> = data.iter().map(|&x| {
        let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + inner.tanh())
    }).collect();

    let shape = tape.values[input].shape.clone();
    tape.variable(result, shape)
}

/// Scaled dot-product attention: softmax(Q·K^T / √d_k) · V
///
/// Q: [batch, seq_q, d_k]
/// K: [batch, seq_k, d_k]
/// V: [batch, seq_k, d_v]
/// Output: [batch, seq_q, d_v]
///
/// For simplicity, this implementation handles single-batch, single-head.
/// Multi-head is built by splitting d_model into n_heads × d_k.
pub fn scaled_dot_product_attention(
    tape: &mut Tape,
    q: usize,
    k: usize,
    v: usize,
    d_k: usize,
) -> usize {
    // Q·K^T
    let q_data = tape.value(q).to_vec();
    let k_data = tape.value(k).to_vec();
    let v_data = tape.value(v).to_vec();

    let seq_q = tape.values[q].shape[0];
    let seq_k = tape.values[k].shape[0];
    let d_v = tape.values[v].shape[1];

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let mut scores = vec![0.0f32; seq_q * seq_k];

    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut dot = 0.0f32;
            for d in 0..d_k {
                dot += q_data[i * d_k + d] * k_data[j * d_k + d];
            }
            scores[i * seq_k + j] = dot * scale;
        }
    }

    // Softmax over seq_k dimension (for each query position)
    for i in 0..seq_q {
        let offset = i * seq_k;
        let max = scores[offset..offset + seq_k].iter().cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..seq_k {
            scores[offset + j] = (scores[offset + j] - max).exp();
            sum += scores[offset + j];
        }
        for j in 0..seq_k {
            scores[offset + j] /= sum;
        }
    }

    // Attention output: scores @ V
    let mut output = vec![0.0f32; seq_q * d_v];
    for i in 0..seq_q {
        for j in 0..d_v {
            let mut sum = 0.0f32;
            for k_idx in 0..seq_k {
                sum += scores[i * seq_k + k_idx] * v_data[k_idx * d_v + j];
            }
            output[i * d_v + j] = sum;
        }
    }

    tape.variable(output, vec![seq_q, d_v])
}

/// Multi-head attention.
///
/// Splits d_model into n_heads × d_k, applies attention per head,
/// then concatenates and projects.
pub fn multi_head_attention(
    tape: &mut Tape,
    q_input: usize,
    k_input: usize,
    v_input: usize,
    w_q: usize,  // [d_model, d_model]
    w_k: usize,  // [d_model, d_model]
    w_v: usize,  // [d_model, d_model]
    w_o: usize,  // [d_model, d_model]
    n_heads: usize,
) -> usize {
    let d_model = tape.values[q_input].shape[1];
    let d_k = d_model / n_heads;
    let seq_len = tape.values[q_input].shape[0];

    // Project Q, K, V
    let q_proj = tape.matmul(q_input, w_q);
    let k_proj = tape.matmul(k_input, w_k);
    let v_proj = tape.matmul(v_input, w_v);

    // For Phase 1: single-head attention on the full projection
    // Phase 2 would split into heads and process in parallel
    let attn_out = scaled_dot_product_attention(tape, q_proj, k_proj, v_proj, d_model);

    // Output projection
    tape.matmul(attn_out, w_o)
}

/// Feed-forward network: FFN(x) = GELU(x·W1 + b1)·W2 + b2
pub fn feed_forward(
    tape: &mut Tape,
    input: usize,
    w1: usize,  // [d_model, d_ff]
    w2: usize,  // [d_ff, d_model]
) -> usize {
    let h = tape.matmul(input, w1);
    let activated = gelu(tape, h);
    tape.matmul(activated, w2)
}

/// Single Transformer encoder layer:
///   x → LayerNorm → MultiHeadAttn → Residual → LayerNorm → FFN → Residual
pub fn transformer_encoder_layer(
    tape: &mut Tape,
    input: usize,
    w_q: usize, w_k: usize, w_v: usize, w_o: usize,
    ff_w1: usize, ff_w2: usize,
    n_heads: usize,
    d_model: usize,
) -> usize {
    // Pre-norm attention
    let normed1 = layer_norm(tape, input, d_model, 1e-5);
    let attn = multi_head_attention(tape, normed1, normed1, normed1, w_q, w_k, w_v, w_o, n_heads);
    let residual1 = tape.add(input, attn);

    // Pre-norm FFN
    let normed2 = layer_norm(tape, residual1, d_model, 1e-5);
    let ff_out = feed_forward(tape, normed2, ff_w1, ff_w2);
    tape.add(residual1, ff_out)
}

/// Sinusoidal positional encoding.
pub fn positional_encoding(seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; seq_len * d_model];

    for pos in 0..seq_len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10000.0f32.powf(2.0 * i as f32 / d_model as f32);
            pe[pos * d_model + 2 * i] = angle.sin();
            pe[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }

    pe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let mut tape = Tape::new();
        let input = tape.variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);

        let normed = layer_norm(&mut tape, input, 4, 1e-5);
        let result = tape.value(normed);

        // Each row should have mean ≈ 0 and std ≈ 1
        let mean: f32 = result[0..4].iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {mean}");

        let var: f32 = result[0..4].iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.1, "Var should be ~1, got {var}");
    }

    #[test]
    fn test_gelu() {
        let mut tape = Tape::new();
        let input = tape.variable(vec![0.0, 1.0, -1.0, 2.0], vec![4]);

        let activated = gelu(&mut tape, input);
        let result = tape.value(activated);

        // GELU(0) = 0
        assert!(result[0].abs() < 1e-5);
        // GELU(1) ≈ 0.8413
        assert!((result[1] - 0.8413).abs() < 0.01);
        // GELU(-1) ≈ -0.1587
        assert!((result[2] - (-0.1587)).abs() < 0.01);
    }

    #[test]
    fn test_attention() {
        let mut tape = Tape::new();

        // Q, K, V: [4, 8] (seq_len=4, d_k=8)
        let q = tape.variable(vec![0.1; 32], vec![4, 8]);
        let k = tape.variable(vec![0.1; 32], vec![4, 8]);
        let v = tape.variable(vec![0.5; 32], vec![4, 8]);

        let out = scaled_dot_product_attention(&mut tape, q, k, v, 8);
        let result = tape.value(out);

        // Output shape should be [4, 8]
        assert_eq!(result.len(), 32);
        // Values should be close to V values since uniform attention
        assert!((result[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding(10, 64);
        assert_eq!(pe.len(), 10 * 64);

        // First position sin should be 0
        assert!(pe[0].abs() < 1e-5);
        // First position cos should be 1
        assert!((pe[1] - 1.0).abs() < 1e-5);

        // Different positions should have different encodings
        let pos0 = &pe[0..64];
        let pos1 = &pe[64..128];
        let diff: f32 = pos0.iter().zip(pos1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "Positional encodings should differ");
    }

    #[test]
    fn test_transformer_layer() {
        let mut tape = Tape::new();

        let seq_len = 4;
        let d_model = 8;
        let d_ff = 16;
        let n_heads = 2;

        // Input: [seq_len, d_model]
        let input = tape.variable(vec![0.1; seq_len * d_model], vec![seq_len, d_model]);

        // Weight matrices (small random values)
        let make_weights = |tape: &mut Tape, rows: usize, cols: usize| -> usize {
            let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
            tape.variable(data, vec![rows, cols])
        };

        let w_q = make_weights(&mut tape, d_model, d_model);
        let w_k = make_weights(&mut tape, d_model, d_model);
        let w_v = make_weights(&mut tape, d_model, d_model);
        let w_o = make_weights(&mut tape, d_model, d_model);
        let ff_w1 = make_weights(&mut tape, d_model, d_ff);
        let ff_w2 = make_weights(&mut tape, d_ff, d_model);

        let output = transformer_encoder_layer(
            &mut tape, input,
            w_q, w_k, w_v, w_o,
            ff_w1, ff_w2,
            n_heads, d_model,
        );

        let result = tape.value(output);
        assert_eq!(result.len(), seq_len * d_model);

        // Output should be finite
        assert!(result.iter().all(|x| x.is_finite()));
    }
}
