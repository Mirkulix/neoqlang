use qo_values::ValueScores;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mood {
    Learning,
    Focused,
    Restless,
    Creating,
    Reflecting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub mood: Mood,
    pub energy: f32,
    pub heartbeat: u64,
    pub agents_active: u8,
    pub agents_idle: u8,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub values: ValueScores,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            mood: Mood::Learning,
            energy: 100.0,
            heartbeat: 0,
            agents_active: 0,
            agents_idle: 6,
            tasks_completed: 0,
            tasks_failed: 0,
            values: ValueScores::default(),
        }
    }
}

impl ConsciousnessState {
    /// Increment heartbeat and regen energy when agents are idle.
    pub fn tick(&mut self) {
        self.heartbeat += 1;
        if self.agents_active == 0 {
            self.energy = (self.energy + 2.0).min(100.0);
        }
    }

    /// Drain energy by amount. If energy drops below 20, switch to Restless.
    pub fn drain_energy(&mut self, amount: f32) {
        self.energy = (self.energy - amount).max(0.0);
        if self.energy < 20.0 {
            self.mood = Mood::Restless;
        }
    }

    /// Record a completed task.
    pub fn task_completed(&mut self) {
        self.tasks_completed += 1;
    }

    /// Record a failed task.
    pub fn task_failed(&mut self) {
        self.tasks_failed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state() {
        let state = ConsciousnessState::default();
        assert_eq!(state.mood, Mood::Learning);
        assert_eq!(state.energy, 100.0);
        assert_eq!(state.heartbeat, 0);
        assert_eq!(state.agents_idle, 6);
    }

    #[test]
    fn tick_increments_heartbeat() {
        let mut state = ConsciousnessState::default();
        state.tick();
        assert_eq!(state.heartbeat, 1);
        state.tick();
        assert_eq!(state.heartbeat, 2);
    }

    #[test]
    fn drain_energy_sets_restless() {
        let mut state = ConsciousnessState::default();
        // Drain to below 20
        state.drain_energy(85.0);
        assert!(state.energy < 20.0);
        assert_eq!(state.mood, Mood::Restless);
    }

    #[test]
    fn energy_regens_on_idle_tick() {
        let mut state = ConsciousnessState::default();
        state.energy = 50.0;
        state.agents_active = 0;
        state.tick();
        assert!(state.energy > 50.0);
    }
}
