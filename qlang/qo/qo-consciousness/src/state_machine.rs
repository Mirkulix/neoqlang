use qo_values::{Value, ValueScores};
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

/// Input events that drive state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateEvent {
    ChatReceived,
    TaskCompleted { agent: String },
    TaskFailed { agent: String, error: String },
    GoalCreated { description: String },
    GoalCompleted { description: String },
    GoalFailed { description: String },
    ValueCheck { scores: [f32; 5] },
    Idle,
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

    /// Process a StateEvent and update mood/values accordingly.
    pub fn process_event(&mut self, event: &StateEvent) {
        match event {
            StateEvent::ChatReceived => {
                self.mood = Mood::Focused;
                self.drain_energy(1.0);
            }
            StateEvent::TaskCompleted { .. } => {
                self.tasks_completed += 1;
                self.mood = Mood::Creating;
            }
            StateEvent::TaskFailed { .. } => {
                self.tasks_failed += 1;
                if self.tasks_failed > self.tasks_completed / 2 {
                    self.mood = Mood::Restless;
                }
            }
            StateEvent::GoalCreated { .. } => {
                self.mood = Mood::Focused;
            }
            StateEvent::GoalCompleted { .. } => {
                self.mood = Mood::Reflecting;
                let current = self.values.get(Value::Entwicklung);
                self.values.set(Value::Entwicklung, current + 0.02);
            }
            StateEvent::GoalFailed { .. } => {
                self.mood = Mood::Restless;
                self.drain_energy(5.0);
            }
            StateEvent::ValueCheck { scores } => {
                self.values.achtsamkeit = scores[0].clamp(0.0, 1.0);
                self.values.anerkennung = scores[1].clamp(0.0, 1.0);
                self.values.aufmerksamkeit = scores[2].clamp(0.0, 1.0);
                self.values.entwicklung = scores[3].clamp(0.0, 1.0);
                self.values.sinn = scores[4].clamp(0.0, 1.0);
            }
            StateEvent::Idle => {
                if self.energy > 80.0 {
                    self.mood = Mood::Learning;
                } else if self.energy < 30.0 {
                    self.mood = Mood::Restless;
                }
            }
        }
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

    #[test]
    fn test_process_chat_event() {
        let mut state = ConsciousnessState::default();
        state.process_event(&StateEvent::ChatReceived);
        assert_eq!(state.mood, Mood::Focused);
        assert!(state.energy < 100.0);
    }

    #[test]
    fn test_process_goal_completed() {
        let mut state = ConsciousnessState::default();
        let before = state.values.get(Value::Entwicklung);
        state.process_event(&StateEvent::GoalCompleted {
            description: "Finish phase 1".to_string(),
        });
        assert_eq!(state.mood, Mood::Reflecting);
        let after = state.values.get(Value::Entwicklung);
        assert!(after > before, "Entwicklung should increase after GoalCompleted");
        assert!((after - (before + 0.02)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_failures_set_restless() {
        let mut state = ConsciousnessState::default();
        // Two failures, zero completions → tasks_failed (2) > tasks_completed (0) / 2 (0)
        state.process_event(&StateEvent::TaskFailed {
            agent: "agent-1".to_string(),
            error: "timeout".to_string(),
        });
        assert_eq!(state.mood, Mood::Restless);
        state.process_event(&StateEvent::TaskFailed {
            agent: "agent-2".to_string(),
            error: "crash".to_string(),
        });
        assert_eq!(state.mood, Mood::Restless);
        assert_eq!(state.tasks_failed, 2);
    }

    #[test]
    fn test_idle_transitions() {
        let mut state = ConsciousnessState::default();
        // High energy → Learning
        state.energy = 90.0;
        state.mood = Mood::Focused;
        state.process_event(&StateEvent::Idle);
        assert_eq!(state.mood, Mood::Learning);

        // Low energy → Restless
        state.energy = 20.0;
        state.mood = Mood::Focused;
        state.process_event(&StateEvent::Idle);
        assert_eq!(state.mood, Mood::Restless);

        // Mid energy → no change
        state.energy = 50.0;
        state.mood = Mood::Creating;
        state.process_event(&StateEvent::Idle);
        assert_eq!(state.mood, Mood::Creating);
    }

    #[test]
    fn test_tick_regens_only_when_idle() {
        let mut state = ConsciousnessState::default();
        state.energy = 50.0;
        state.agents_active = 2;
        state.tick();
        // Energy should NOT regen when agents_active > 0
        assert_eq!(state.energy, 50.0);
        assert_eq!(state.heartbeat, 1);
    }

    #[test]
    fn test_energy_never_exceeds_100() {
        let mut state = ConsciousnessState::default();
        state.energy = 99.9;
        state.agents_active = 0;
        state.tick();
        assert!(state.energy <= 100.0);
    }

    #[test]
    fn test_mood_lifecycle() {
        let mut state = ConsciousnessState::default();
        // Start in Learning
        assert_eq!(state.mood, Mood::Learning);
        // ChatReceived → Focused
        state.process_event(&StateEvent::ChatReceived);
        assert_eq!(state.mood, Mood::Focused);
        // TaskCompleted → Creating
        state.process_event(&StateEvent::TaskCompleted { agent: "dev".to_string() });
        assert_eq!(state.mood, Mood::Creating);
        // GoalCompleted → Reflecting
        state.process_event(&StateEvent::GoalCompleted { description: "done".to_string() });
        assert_eq!(state.mood, Mood::Reflecting);
        // GoalFailed → Restless
        state.process_event(&StateEvent::GoalFailed { description: "oops".to_string() });
        assert_eq!(state.mood, Mood::Restless);
    }

    #[test]
    fn test_tasks_counter_increments() {
        let mut state = ConsciousnessState::default();
        assert_eq!(state.tasks_completed, 0);
        state.task_completed();
        state.task_completed();
        state.task_completed();
        assert_eq!(state.tasks_completed, 3);
    }

    #[test]
    fn test_process_idle_high_energy() {
        let mut state = ConsciousnessState::default();
        state.energy = 90.0;
        state.mood = Mood::Focused;
        state.process_event(&StateEvent::Idle);
        assert_eq!(state.mood, Mood::Learning);
    }
}
