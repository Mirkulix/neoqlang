use serde::{Deserialize, Serialize};

/// Simplified quantum state for QO's evolution.
/// Represents a probability distribution over strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// Probability weights for different strategies
    pub strategy_weights: Vec<f32>,
    /// Strategy labels
    pub strategies: Vec<String>,
    /// Generation counter
    pub generation: u64,
    /// Von Neumann entropy (measure of uncertainty)
    pub entropy: f32,
}

impl QuantumState {
    pub fn new(strategies: Vec<String>) -> Self {
        let n = strategies.len();
        let uniform = 1.0 / n as f32;
        let weights = vec![uniform; n];
        let entropy = -(uniform * uniform.ln() * n as f32);
        Self {
            strategy_weights: weights,
            strategies,
            generation: 0,
            entropy,
        }
    }

    /// Evolve the state based on observed outcome.
    /// strategy_index: which strategy was used
    /// success: whether it succeeded (positive reward) or failed (negative)
    /// learning_rate: how much to update (0.0 = no change, 1.0 = full shift)
    pub fn evolve(&mut self, strategy_index: usize, success: bool, learning_rate: f32) {
        if strategy_index >= self.strategy_weights.len() { return; }

        let n = self.strategy_weights.len();
        let reward = if success { learning_rate } else { -learning_rate * 0.5 };

        // Update weights (softmax-style)
        self.strategy_weights[strategy_index] += reward;

        // Normalize to probability distribution
        let min_val = self.strategy_weights.iter().cloned().fold(f32::INFINITY, f32::min);
        if min_val < 0.0 {
            for w in &mut self.strategy_weights {
                *w -= min_val;
            }
        }
        let sum: f32 = self.strategy_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.strategy_weights {
                *w /= sum;
            }
        } else {
            let uniform = 1.0 / n as f32;
            self.strategy_weights = vec![uniform; n];
        }

        // Update entropy
        self.entropy = self.strategy_weights.iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum();

        self.generation += 1;
    }

    /// "Measure" the quantum state — select the most probable strategy.
    pub fn measure(&self) -> Option<(usize, &str)> {
        self.strategy_weights.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| (i, self.strategies[i].as_str()))
    }

    /// Get a summary for API responses
    pub fn summary(&self) -> QuantumSummary {
        QuantumSummary {
            generation: self.generation,
            entropy: self.entropy,
            top_strategy: self.measure().map(|(_, s)| s.to_string()),
            strategies: self.strategies.iter().zip(self.strategy_weights.iter())
                .map(|(s, w)| (s.clone(), *w))
                .collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumSummary {
    pub generation: u64,
    pub entropy: f32,
    pub top_strategy: Option<String>,
    pub strategies: Vec<(String, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evolve_shifts_weights() {
        let mut qs = QuantumState::new(vec!["A".into(), "B".into(), "C".into()]);
        qs.evolve(0, true, 0.1);
        assert!(qs.strategy_weights[0] > qs.strategy_weights[1]);
    }

    #[test]
    fn measure_returns_best() {
        let mut qs = QuantumState::new(vec!["Alpha".into(), "Beta".into()]);
        qs.evolve(1, true, 0.5);
        qs.evolve(1, true, 0.5);
        let (idx, name) = qs.measure().unwrap();
        assert_eq!(idx, 1);
        assert_eq!(name, "Beta");
    }

    #[test]
    fn generation_increments() {
        let mut qs = QuantumState::new(vec!["X".into()]);
        assert_eq!(qs.generation, 0);
        qs.evolve(0, true, 0.1);
        assert_eq!(qs.generation, 1);
    }

    #[test]
    fn entropy_decreases_with_learning() {
        let mut qs = QuantumState::new(vec!["A".into(), "B".into(), "C".into()]);
        let initial_entropy = qs.entropy;
        // Repeatedly reward strategy 0
        for _ in 0..20 {
            qs.evolve(0, true, 0.1);
        }
        // Entropy should decrease (more certainty)
        assert!(qs.entropy < initial_entropy);
    }

    #[test]
    fn weights_stay_normalized() {
        let mut qs = QuantumState::new(vec!["X".into(), "Y".into()]);
        for i in 0..100 {
            qs.evolve(i % 2, i % 3 == 0, 0.2);
        }
        let sum: f32 = qs.strategy_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
