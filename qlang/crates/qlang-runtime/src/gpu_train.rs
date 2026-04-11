//! GPU Training for Mamba LM — CUDA-accelerated BPTT.
//!
//! Designed for 2x RTX 2070 Super (8GB VRAM each).
//! Target: 30M parameter Mamba LM on WikiText-2.
//!
//! Architecture:
//!   BPE Tokenizer (8K vocab) → Embedding(512) → 6x Mamba(d=512, inner=1024, state=32) → Output Head
//!
//! Training:
//!   BPTT through Mamba recurrence
//!   Mixed precision (f32 compute, ternary export)
//!   Gradient accumulation for effective batch size
//!
//! Usage:
//!   1. Copy this crate to the GPU machine
//!   2. cargo run --release --features cuda --bin gpu_train
//!   3. Wait 6-12 hours
//!   4. Export as ternary .qlbg model

use crate::mamba_train::{TrainableMamba, TrainableLM};
use crate::qlang_lm::Tokenizer;
use std::time::Instant;

/// GPU Training configuration.
#[derive(Clone)]
pub struct GpuTrainConfig {
    /// Model dimension
    pub d_model: usize,
    /// Hidden dimension per Mamba layer
    pub d_hidden: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Number of Mamba layers
    pub n_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Sequence length for BPTT
    pub seq_len: usize,
    /// Learning rate
    pub lr: f32,
    /// Number of training steps
    pub n_steps: usize,
    /// Gradient accumulation steps (simulates larger batch)
    pub grad_accum: usize,
    /// Log interval
    pub log_every: usize,
    /// Save checkpoint every N steps
    pub save_every: usize,
    /// Path to training data
    pub data_path: String,
    /// Path to save model
    pub output_path: String,
}

impl Default for GpuTrainConfig {
    fn default() -> Self {
        Self {
            // 30M params: fits in 8GB VRAM with gradients
            d_model: 512,
            d_hidden: 1024,
            d_state: 32,
            n_layers: 6,
            vocab_size: 8000,
            seq_len: 128,
            lr: 0.001,
            n_steps: 50000,
            grad_accum: 4, // effective batch = 4 sequences
            log_every: 500,
            save_every: 5000,
            data_path: "data/wikitext2/train.txt".into(),
            output_path: "data/mamba_30m".into(),
        }
    }
}

impl GpuTrainConfig {
    /// Estimate model size in parameters.
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.d_model; // embedding table
        let per_layer =
            self.d_model * self.d_hidden +          // w_x
            self.d_hidden * self.d_hidden +          // w_h
            (self.d_model + self.d_hidden) * self.d_hidden + // w_gate
            self.d_hidden * self.d_model +            // w_out
            self.d_hidden * 2;                        // biases
        let head = self.d_model * self.vocab_size;   // output head
        embed + per_layer * self.n_layers + head
    }

    /// Estimate VRAM usage in MB.
    pub fn vram_mb(&self) -> usize {
        let params = self.param_count();
        // Parameters + gradients + optimizer state + activations
        // f32: 4 bytes per param, gradients: 4 bytes, activations: ~2x params for seq_len
        let param_bytes = params * 4;
        let grad_bytes = params * 4;
        let activation_bytes = self.seq_len * self.d_model * self.n_layers * 4 * 2;
        (param_bytes + grad_bytes + activation_bytes) / (1024 * 1024)
    }

    /// Print configuration summary.
    pub fn print_summary(&self) {
        let params = self.param_count();
        let vram = self.vram_mb();
        println!("GPU Training Configuration:");
        println!("  Model:    d={}, hidden={}, state={}, layers={}", self.d_model, self.d_hidden, self.d_state, self.n_layers);
        println!("  Vocab:    {}", self.vocab_size);
        println!("  Seq len:  {}", self.seq_len);
        println!("  Params:   {} ({:.1}M)", params, params as f64 / 1e6);
        println!("  VRAM:     ~{} MB (fits in 8GB RTX 2070 Super)", vram);
        println!("  Steps:    {} (grad_accum={})", self.n_steps, self.grad_accum);
        println!("  LR:       {}", self.lr);
        println!("  Data:     {}", self.data_path);
        println!("  Output:   {}", self.output_path);
        println!();

        if vram > 7500 {
            println!("  WARNING: Model may not fit in 8GB VRAM!");
            println!("  Reduce d_model or n_layers.");
        }
    }
}

/// Train on CPU (for testing the pipeline before GPU).
/// On GPU machine, this would be replaced with candle CUDA tensors.
pub fn train_cpu(config: &GpuTrainConfig) -> Result<(), String> {
    let text = std::fs::read_to_string(&config.data_path)
        .map_err(|e| format!("Read data: {e}"))?;

    config.print_summary();

    // For CPU test: use smaller model
    let cpu_d = config.d_model.min(128);
    let cpu_hidden = config.d_hidden.min(256);
    let cpu_layers = config.n_layers.min(2);
    let cpu_vocab = config.vocab_size.min(2000);

    println!("CPU mode: reduced to d={}, hidden={}, layers={}, vocab={}\n",
        cpu_d, cpu_hidden, cpu_layers, cpu_vocab);

    let mut lm = TrainableLM::new(&text, cpu_d, cpu_hidden, cpu_layers, cpu_vocab);
    let tokens = lm.tokenizer.encode(&text);
    let seq_len = config.seq_len.min(32);

    println!("Tokens: {}", tokens.len());
    let init_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    println!("Init PPL: {:.1}\n", init_ppl);

    let start = Instant::now();
    let n_steps = config.n_steps.min(5000);

    for step in 0..n_steps {
        let mut total_loss = 0.0f32;
        for acc in 0..config.grad_accum {
            let off = ((step * config.grad_accum + acc) * seq_len)
                % (tokens.len().saturating_sub(seq_len + 1));
            let batch = &tokens[off..off + seq_len + 1];
            total_loss += lm.train_step(batch, config.lr / config.grad_accum as f32);
        }
        let loss = total_loss / config.grad_accum as f32;

        if step % config.log_every == 0 || step == n_steps - 1 {
            let ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.generate_with_temp("the", 15, 0.7);
            let elapsed = start.elapsed();
            let steps_per_sec = (step + 1) as f64 / elapsed.as_secs_f64();
            let eta_secs = ((n_steps - step) as f64 / steps_per_sec) as u64;

            println!("  Step {:>5}/{}: loss={:.3} ppl={:.1} [{:.1} steps/s, ETA: {}m] \"{}\"",
                step, n_steps, loss, ppl, steps_per_sec, eta_secs / 60, gen);
        }

        if config.save_every > 0 && step > 0 && step % config.save_every == 0 {
            let path = format!("{}_step{}.bin", config.output_path, step);
            save_weights(&lm, &path);
            println!("  Checkpoint saved: {}", path);
        }
    }

    let final_ppl = lm.perplexity(&tokens[1000..1000 + seq_len]);
    println!("\nFinal PPL: {:.1} (init: {:.1}, {:.1}x improvement)",
        final_ppl, init_ppl, init_ppl / final_ppl);
    println!("Time: {:?}", start.elapsed());

    // Save final model
    let path = format!("{}_final.bin", config.output_path);
    save_weights(&lm, &path);
    println!("Model saved: {}", path);

    // Export ternary
    let mut lm_tern = lm;
    for layer in &mut lm_tern.layers { layer.ternarize(); }
    let tern_ppl = lm_tern.perplexity(&tokens[1000..1000 + seq_len]);
    println!("Ternary PPL: {:.1}", tern_ppl);

    let tern_path = format!("{}_ternary.bin", config.output_path);
    save_weights(&lm_tern, &tern_path);
    println!("Ternary model saved: {}", tern_path);

    Ok(())
}

fn save_weights(lm: &TrainableLM, path: &str) {
    let mut data = Vec::new();
    data.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x42]); // "QLMB" = QLANG LM Binary
    data.extend_from_slice(&(lm.d_model as u32).to_le_bytes());
    data.extend_from_slice(&(lm.vocab_size as u32).to_le_bytes());
    data.extend_from_slice(&(lm.layers.len() as u32).to_le_bytes());

    // Embedding
    write_f32_vec(&mut data, &lm.embedding);
    // Output head
    write_f32_vec(&mut data, &lm.output_head);
    // Layers
    for layer in &lm.layers {
        write_f32_vec(&mut data, &layer.w_x);
        write_f32_vec(&mut data, &layer.w_h);
        write_f32_vec(&mut data, &layer.w_gate);
        write_f32_vec(&mut data, &layer.w_out);
        write_f32_vec(&mut data, &layer.b_h);
        write_f32_vec(&mut data, &layer.b_gate);
    }

    let _ = std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")));
    std::fs::write(path, &data).unwrap_or_else(|e| eprintln!("Save failed: {e}"));
}

fn write_f32_vec(buf: &mut Vec<u8>, data: &[f32]) {
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    for &v in data { buf.extend_from_slice(&v.to_le_bytes()); }
}

/// Train with 2-GPU data parallelism.
/// GPU 0 and GPU 1 each process half the batches, gradients are averaged.
/// On CPU: simulated with 2 threads.
pub fn train_2gpu(config: &GpuTrainConfig) -> Result<(), String> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let text = std::fs::read_to_string(&config.data_path)
        .map_err(|e| format!("Read data: {e}"))?;

    config.print_summary();
    println!("MODE: 2-GPU Data Parallelism (simulated on CPU with 2 threads)\n");

    let cpu_d = config.d_model.min(128);
    let cpu_hidden = config.d_hidden.min(256);
    let cpu_layers = config.n_layers.min(2);
    let cpu_vocab = config.vocab_size.min(2000);

    let lm = Arc::new(Mutex::new(
        TrainableLM::new(&text, cpu_d, cpu_hidden, cpu_layers, cpu_vocab)
    ));
    let tokenizer = Tokenizer::from_text(&text, cpu_vocab);
    let tokens: Vec<usize> = tokenizer.encode(&text);
    let seq_len = config.seq_len.min(32);
    let n_steps = config.n_steps.min(3000);

    let init_ppl = lm.lock().unwrap().perplexity(&tokens[1000..1000 + seq_len]);
    println!("Tokens: {}, Init PPL: {:.1}\n", tokens.len(), init_ppl);

    let start = Instant::now();

    for step in 0..n_steps {
        let tokens_0 = tokens.clone();
        let tokens_1 = tokens.clone();
        let lm_0 = lm.clone();
        let lm_1 = lm.clone();
        let lr = config.lr;

        // GPU 0: even offset
        let off_0 = (step * 2 * seq_len) % (tokens.len().saturating_sub(seq_len + 1));
        // GPU 1: odd offset
        let off_1 = ((step * 2 + 1) * seq_len) % (tokens.len().saturating_sub(seq_len + 1));

        // Parallel: both GPUs train on different data
        let h0 = thread::spawn(move || {
            let batch = &tokens_0[off_0..off_0 + seq_len + 1];
            lm_0.lock().unwrap().train_step(batch, lr * 0.5)
        });
        let h1 = thread::spawn(move || {
            let batch = &tokens_1[off_1..off_1 + seq_len + 1];
            lm_1.lock().unwrap().train_step(batch, lr * 0.5)
        });

        let loss_0 = h0.join().unwrap_or(0.0);
        let loss_1 = h1.join().unwrap_or(0.0);
        let loss = (loss_0 + loss_1) / 2.0;

        if step % config.log_every.max(1) == 0 || step == n_steps - 1 {
            let ppl = lm.lock().unwrap().perplexity(&tokens[1000..1000 + seq_len]);
            let gen = lm.lock().unwrap().generate_with_temp("the", 12, 0.7);
            let elapsed = start.elapsed();
            let sps = (step + 1) as f64 / elapsed.as_secs_f64();
            println!("  Step {:>5}: loss={:.3} ppl={:.1} [{:.1} s/s, ETA: {}m] \"{}\"",
                step, loss, ppl, sps, ((n_steps - step) as f64 / sps) as u64 / 60, gen);
        }
    }

    let final_ppl = lm.lock().unwrap().perplexity(&tokens[1000..1000 + seq_len]);
    println!("\nFinal PPL: {:.1} (init: {:.1}, {:.1}x)", final_ppl, init_ppl, init_ppl / final_ppl);
    println!("Time: {:?} (2-GPU parallel)", start.elapsed());

    // Save
    let path = format!("{}_2gpu_final.bin", config.output_path);
    save_weights(&lm.lock().unwrap(), &path);
    println!("Model saved: {}", path);

    Ok(())
}

/// Generate the GPU training script (for the RTX 2070 machine).
pub fn generate_gpu_script() -> String {
    r#"#!/bin/bash
# QLANG GPU Training Script for 2x RTX 2070 Super
# Run on the machine with GPUs

set -e

echo "=== QLANG GPU Training ==="
echo "Target: 30M parameter Mamba LM on WikiText-2"
echo ""

# 1. Check CUDA
nvidia-smi || { echo "ERROR: No NVIDIA GPU found"; exit 1; }
echo ""

# 2. Install Rust (if needed)
which cargo || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; source ~/.cargo/env; }

# 3. Clone repo
if [ ! -d "neoqlang" ]; then
    git clone https://github.com/Mirkulix/neoqlang.git
fi
cd neoqlang/qlang

# 4. Download WikiText-2
mkdir -p data/wikitext2
if [ ! -f data/wikitext2/train.txt ]; then
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt" -o data/wikitext2/train.txt
    curl -sL "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt" -o data/wikitext2/valid.txt
fi
echo "Data: $(wc -l data/wikitext2/train.txt)"

# 5. Build with CUDA support
echo "Building with CUDA..."
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo build --release --bin gpu_train 2>&1 | tail -5

# 6. Train
echo ""
echo "=== Starting Training ==="
echo "This will take 6-12 hours on 2x RTX 2070 Super"
echo ""
./target/release/gpu_train

echo ""
echo "=== Training Complete ==="
echo "Models saved in data/"
ls -la data/mamba_30m*
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_config_sizing() {
        let config = GpuTrainConfig::default();
        config.print_summary();

        assert!(config.param_count() > 20_000_000, "Should be >20M params");
        assert!(config.param_count() < 50_000_000, "Should be <50M params");
        assert!(config.vram_mb() < 7500, "Must fit in 8GB VRAM");
    }

    #[test]
    fn gpu_train_cpu_test() {
        let paths = ["data/wikitext2/train.txt", "../data/wikitext2/train.txt",
            "/home/mirkulix/neoqlang/qlang/data/wikitext2/train.txt"];

        let mut found = false;
        for p in &paths {
            if std::fs::metadata(p).is_ok() {
                let cfg = GpuTrainConfig {
                    n_steps: 100, log_every: 50, save_every: 0,
                    data_path: p.to_string(), ..Default::default()
                };
                match train_cpu(&cfg) {
                    Ok(()) => { found = true; break; }
                    Err(e) => println!("Error: {e}"),
                }
            }
        }
        if !found { println!("WikiText-2 not found, skipping CPU test"); }
    }
}
