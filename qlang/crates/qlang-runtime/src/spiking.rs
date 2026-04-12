//! Spiking Neural Network — bio-inspired neuromorphic computing.
//!
//! Implements Leaky Integrate-and-Fire (LIF) neurons with Spike-Timing
//! Dependent Plasticity (STDP) learning and ternary weights.
//!
//! Architecture:
//!   Input → [Rate Encoding] → SpikingLayer(s) → [Spike Counts] → Classification
//!                                                       ↓
//!                                              [spikes_to_hd] → HdMemory
//!
//! Design principles:
//! - No float multiplication in the spike processing hot loop (neuromorphic)
//! - Ternary weights only: {-1, 0, +1}
//! - STDP extends Hebbian learning with temporal dynamics
//! - SynapseHD pattern: SNN features → HDC classification

use crate::hdc::HdVector;

// ---------------------------------------------------------------------------
// XorShift RNG (consistent with existing codebase)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

// ---------------------------------------------------------------------------
// LIF Neuron
// ---------------------------------------------------------------------------

/// Leaky Integrate-and-Fire neuron.
///
/// Models membrane potential dynamics:
///   V(t+dt) = V(t) + dt/tau * (-V(t) + I(t))
///
/// Fires a spike when V >= threshold, then resets.
#[derive(Clone, Debug)]
pub struct LIFNeuron {
    /// Current membrane potential V(t).
    pub membrane: f32,
    /// Spike threshold (default 1.0).
    pub threshold: f32,
    /// Reset potential after spike (default 0.0).
    pub reset: f32,
    /// Membrane time constant in ms (default 20.0).
    pub tau: f32,
    /// Refractory period in timesteps (default 5.0).
    pub refractory: f32,
    /// Remaining refractory time.
    refractory_counter: f32,
}

impl LIFNeuron {
    pub fn new() -> Self {
        Self {
            membrane: 0.0,
            threshold: 1.0,
            reset: 0.0,
            tau: 20.0,
            refractory: 5.0,
            refractory_counter: 0.0,
        }
    }

    /// Advance one timestep. Returns true if the neuron spikes.
    ///
    /// No multiplication in the caller's hot loop — the division by tau
    /// is internal to the neuron model (not in the weight*spike path).
    pub fn step(&mut self, input_current: f32, dt: f32) -> bool {
        if self.refractory_counter > 0.0 {
            self.refractory_counter -= dt;
            return false;
        }
        // Leaky integration: V(t+dt) = V(t) + dt/tau * (-V(t) + I(t))
        self.membrane += (dt / self.tau) * (-self.membrane + input_current);
        if self.membrane >= self.threshold {
            self.membrane = self.reset;
            self.refractory_counter = self.refractory;
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// STDP
// ---------------------------------------------------------------------------

/// Spike-Timing Dependent Plasticity parameters.
#[derive(Clone, Debug)]
pub struct STDPParams {
    /// LTP amplitude (default 0.01).
    pub a_plus: f32,
    /// LTD amplitude (default 0.012).
    pub a_minus: f32,
    /// LTP time constant (default 20.0).
    pub tau_plus: f32,
    /// LTD time constant (default 20.0).
    pub tau_minus: f32,
}

impl Default for STDPParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
        }
    }
}

/// Compute STDP weight change from spike timing difference.
///
/// `dt` = t_pre - t_post:
/// - dt > 0 (pre before post): LTP (strengthen)
/// - dt < 0 (post before pre): LTD (weaken)
pub fn stdp_update(dt: f32, params: &STDPParams) -> f32 {
    if dt > 0.0 {
        params.a_plus * (-dt / params.tau_plus).exp()
    } else if dt < 0.0 {
        -params.a_minus * (dt / params.tau_minus).exp()
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// SpikingLayer
// ---------------------------------------------------------------------------

/// A layer of LIF neurons with ternary weights and STDP learning.
pub struct SpikingLayer {
    neurons: Vec<LIFNeuron>,
    /// Ternary weights {-1, 0, +1}, row-major [out_dim * in_dim].
    weights: Vec<i8>,
    /// Salience accumulator for ternary weight flips [out_dim * in_dim].
    salience: Vec<f32>,
    /// Last spike time for each neuron: [in_dim] inputs then [out_dim] outputs.
    last_spike_time: Vec<f32>,
    stdp: STDPParams,
    /// Salience threshold for flipping a ternary weight.
    flip_threshold: f32,
    in_dim: usize,
    out_dim: usize,
}

impl SpikingLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = 0xCAFE_BABEu64;
        let weights: Vec<i8> = (0..out_dim * in_dim)
            .map(|_| {
                let v = xorshift(&mut rng) % 3;
                match v { 0 => -1, 1 => 0, _ => 1 }
            })
            .collect();

        Self {
            neurons: (0..out_dim).map(|_| LIFNeuron::new()).collect(),
            weights,
            salience: vec![0.0; out_dim * in_dim],
            last_spike_time: vec![-1000.0; in_dim + out_dim],
            stdp: STDPParams::default(),
            flip_threshold: 0.5,
            in_dim,
            out_dim,
        }
    }

    /// Process one timestep: input spikes -> output spikes.
    ///
    /// The hot loop uses only add/sub on ternary weights (no multiply).
    pub fn step(&mut self, input_spikes: &[bool], t: f32, dt: f32) -> Vec<bool> {
        assert_eq!(input_spikes.len(), self.in_dim);
        let mut output_spikes = vec![false; self.out_dim];

        // Record input spike times
        for i in 0..self.in_dim {
            if input_spikes[i] {
                self.last_spike_time[i] = t;
            }
        }

        for j in 0..self.out_dim {
            // Compute input current from ternary weights x input spikes.
            // Only add/sub — no multiply! The weight is -1, 0, or +1.
            let mut current = 0i32;
            let row = j * self.in_dim;
            for i in 0..self.in_dim {
                if input_spikes[i] {
                    // i8 add/sub only (neuromorphic principle)
                    current += self.weights[row + i] as i32;
                }
            }

            output_spikes[j] = self.neurons[j].step(current as f32, dt);

            // STDP update on output spike
            if output_spikes[j] {
                self.last_spike_time[self.in_dim + j] = t;
                for i in 0..self.in_dim {
                    // Only update synapses where the pre neuron spiked recently
                    let pre_t = self.last_spike_time[i];
                    if pre_t > -999.0 {
                        let dt_spike = t - pre_t; // positive = pre before post = LTP
                        let dw = stdp_update(dt_spike, &self.stdp);
                        self.salience[row + i] += dw;

                        // Flip ternary weight when salience exceeds threshold
                        let s = self.salience[row + i];
                        if s > self.flip_threshold {
                            self.weights[row + i] = 1;
                            self.salience[row + i] = 0.0;
                        } else if s < -self.flip_threshold {
                            self.weights[row + i] = -1;
                            self.salience[row + i] = 0.0;
                        }
                    }
                }
            }
        }
        output_spikes
    }

    /// Read the ternary weight matrix (for inspection/testing).
    pub fn weights(&self) -> &[i8] {
        &self.weights
    }
}

// ---------------------------------------------------------------------------
// SpikingNetwork
// ---------------------------------------------------------------------------

/// Multi-layer spiking neural network with rate coding I/O.
pub struct SpikingNetwork {
    layers: Vec<SpikingLayer>,
    /// RNG state for rate coding.
    rng: u64,
}

impl SpikingNetwork {
    /// Create a network from layer sizes, e.g. `&[784, 128, 10]`.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input and output layers");
        let layers = layer_sizes.windows(2)
            .map(|w| SpikingLayer::new(w[0], w[1]))
            .collect();
        Self { layers, rng: 0xDEAD_BEEF }
    }

    /// Encode continuous values as spike train via rate coding.
    /// Higher value -> higher spike probability per timestep.
    pub fn encode_rate(values: &[f32], rng: &mut u64) -> Vec<bool> {
        values.iter().map(|&v| {
            let prob = v.max(0.0).min(1.0);
            let r = xorshift(rng);
            // Compare against threshold: prob * u64::MAX
            let threshold = (prob * u32::MAX as f32) as u64;
            (r & 0xFFFF_FFFF) < threshold
        }).collect()
    }

    /// Run the network for `timesteps` on an input.
    /// Returns spike counts per output neuron.
    pub fn run(&mut self, input: &[f32], timesteps: usize) -> Vec<u32> {
        let out_dim = self.layers.last().unwrap().out_dim;
        let mut counts = vec![0u32; out_dim];
        let dt = 1.0;

        for t in 0..timesteps {
            let time = t as f32;
            // Rate-encode input into spikes
            let mut spikes = Self::encode_rate(input, &mut self.rng);

            // Propagate through layers
            for layer in &mut self.layers {
                spikes = layer.step(&spikes, time, dt);
            }

            // Count output spikes
            for (i, &s) in spikes.iter().enumerate() {
                if s { counts[i] += 1; }
            }
        }
        counts
    }

    /// Classify: run network, return class with most spikes.
    pub fn classify(&mut self, input: &[f32], timesteps: usize, _n_classes: usize) -> usize {
        let counts = self.run(input, timesteps);
        counts.iter().enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Train on labeled data using STDP.
    ///
    /// Presents each sample as a rate-coded spike train for `timesteps`
    /// steps. STDP updates happen automatically during forward propagation.
    pub fn train_stdp(
        &mut self,
        inputs: &[f32],
        _labels: &[u8],
        n_samples: usize,
        input_dim: usize,
        timesteps: usize,
    ) {
        let dt = 1.0;
        for s in 0..n_samples {
            let sample = &inputs[s * input_dim..(s + 1) * input_dim];
            for t in 0..timesteps {
                let time = (s * timesteps + t) as f32;
                let mut spikes = Self::encode_rate(sample, &mut self.rng);
                for layer in &mut self.layers {
                    spikes = layer.step(&spikes, time, dt);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HDC Integration — SynapseHD pattern
// ---------------------------------------------------------------------------

/// Convert spike train output to HDC hypervector for memory storage.
///
/// Each output neuron gets a deterministic HD base vector. The final
/// hypervector is a weighted bundle by spike count. This is the
/// SynapseHD pattern: SNN features -> HDC classification.
pub fn spikes_to_hd(spike_counts: &[u32], dim: usize, seed: u64) -> HdVector {
    if spike_counts.is_empty() {
        return HdVector::zero(dim);
    }

    let mut accum = vec![0i32; dim];
    let mut base_seed = seed;

    for (i, &count) in spike_counts.iter().enumerate() {
        if count == 0 { continue; }
        // Deterministic base vector per neuron index
        let neuron_seed = base_seed.wrapping_add(i as u64).wrapping_mul(6364136223846793005);
        let base = HdVector::random(dim, neuron_seed);
        // Weighted bundle: add count times (integer add only)
        for d in 0..dim {
            accum[d] += base.data[d] as i32 * count as i32;
        }
    }

    // Ternarize
    let data: Vec<i8> = accum.iter().map(|&s| {
        if s > 0 { 1i8 } else if s < 0 { -1i8 } else { 0i8 }
    }).collect();

    HdVector { data, dim }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HdMemory;

    #[test]
    fn lif_neuron_spikes() {
        let mut neuron = LIFNeuron::new();
        neuron.threshold = 0.5;
        neuron.tau = 1.0; // fast integration for test

        // Strong input should cause a spike
        let mut spiked = false;
        for _ in 0..100 {
            if neuron.step(10.0, 1.0) {
                spiked = true;
                break;
            }
        }
        assert!(spiked, "Neuron should fire with strong input current");
    }

    #[test]
    fn lif_refractory() {
        let mut neuron = LIFNeuron::new();
        neuron.threshold = 0.5;
        neuron.tau = 1.0;
        neuron.refractory = 10.0;

        // Drive to spike
        let mut first_spike_t = None;
        for t in 0..50 {
            if neuron.step(10.0, 1.0) {
                if first_spike_t.is_none() {
                    first_spike_t = Some(t);
                } else {
                    // Second spike must be at least refractory period later
                    let gap = t - first_spike_t.unwrap();
                    assert!(
                        gap >= 10,
                        "Neuron fired too soon after refractory: gap={}",
                        gap
                    );
                    return;
                }
            }
        }
        // Should have spiked at least once
        assert!(first_spike_t.is_some(), "Neuron should have spiked");
    }

    #[test]
    fn stdp_ltp_ltd() {
        let params = STDPParams::default();

        // Pre before post (dt > 0): LTP (positive)
        let ltp = stdp_update(5.0, &params);
        assert!(ltp > 0.0, "Pre-before-post should strengthen: got {}", ltp);

        // Post before pre (dt < 0): LTD (negative)
        let ltd = stdp_update(-5.0, &params);
        assert!(ltd < 0.0, "Post-before-pre should weaken: got {}", ltd);

        // Simultaneous (dt = 0): no change
        let zero = stdp_update(0.0, &params);
        assert_eq!(zero, 0.0, "Simultaneous spikes: no change");
    }

    #[test]
    fn spiking_layer_processes_spikes() {
        let mut layer = SpikingLayer::new(8, 4);
        // Increase neuron sensitivity for test
        for n in &mut layer.neurons {
            n.threshold = 0.3;
            n.tau = 1.0;
        }

        let input = vec![true, false, true, true, false, true, false, true];
        let mut any_spike = false;
        for t in 0..50 {
            let out = layer.step(&input, t as f32, 1.0);
            assert_eq!(out.len(), 4);
            if out.iter().any(|&s| s) {
                any_spike = true;
            }
        }
        assert!(any_spike, "Layer should produce at least one output spike");
    }

    #[test]
    fn spiking_network_classifies() {
        // Simple test: two patterns with different rate coding
        let mut net = SpikingNetwork::new(&[4, 8, 2]);
        // Increase sensitivity
        for layer in &mut net.layers {
            for n in &mut layer.neurons {
                n.threshold = 0.3;
                n.tau = 2.0;
            }
        }

        // Pattern A: high on first two dims
        let pattern_a = vec![0.9, 0.9, 0.1, 0.1];
        // Pattern B: high on last two dims
        let pattern_b = vec![0.1, 0.1, 0.9, 0.9];

        let counts_a = net.run(&pattern_a, 100);
        let counts_b = net.run(&pattern_b, 100);

        // Both should produce some output spikes
        let total_a: u32 = counts_a.iter().sum();
        let total_b: u32 = counts_b.iter().sum();
        assert!(
            total_a > 0 || total_b > 0,
            "Network should produce spikes for at least one pattern (a={}, b={})",
            total_a, total_b
        );

        // The classify method should return a valid class
        let class = net.classify(&pattern_a, 100, 2);
        assert!(class < 2, "Class should be 0 or 1, got {}", class);
    }

    #[test]
    fn spikes_to_hd_integration() {
        let counts = vec![5, 0, 3, 1];
        let hv = spikes_to_hd(&counts, 10000, 42);

        assert_eq!(hv.dim, 10000);
        assert!(hv.is_ternary());

        // Non-zero spike counts should produce a non-zero vector
        let nonzero = hv.data.iter().filter(|&&v| v != 0).count();
        assert!(nonzero > 0, "HD vector should be non-zero");

        // Store in HdMemory and retrieve
        let mut mem = HdMemory::new(10000);
        mem.store("spike_pattern_1", hv.clone());

        let (name, sim) = mem.query(&hv).unwrap();
        assert_eq!(name, "spike_pattern_1");
        assert!(sim > 0.9, "Self-query similarity should be high, got {}", sim);

        // Different spike counts should produce a different vector
        let counts2 = vec![0, 4, 0, 6];
        let hv2 = spikes_to_hd(&counts2, 10000, 42);
        let sim2 = hv.similarity(&hv2);
        assert!(
            sim2 < 0.8,
            "Different spike patterns should differ in HD space, sim={}",
            sim2
        );
    }
}
