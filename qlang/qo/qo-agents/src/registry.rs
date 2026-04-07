use crate::agent::{Agent, AgentRole, AgentStatus};
use crate::goal::{Goal, GoalStatus};
use crate::executor;
use qo_llm::LlmRouter;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct AgentSummary {
    pub role: AgentRole,
    pub status: AgentStatus,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
}

pub struct AgentRegistry {
    agents: HashMap<AgentRole, Agent>,
    goals: Vec<Goal>,
    next_goal_id: u64,
}

impl AgentRegistry {
    pub fn new() -> Self {
        let mut agents = HashMap::new();
        for role in AgentRole::ALL {
            agents.insert(role, Agent::new(role));
        }
        Self {
            agents,
            goals: Vec::new(),
            next_goal_id: 1,
        }
    }

    pub fn list_agents(&self) -> Vec<AgentSummary> {
        self.agents
            .values()
            .map(|a| AgentSummary {
                role: a.role,
                status: a.status,
                tasks_completed: a.tasks_completed,
                tasks_failed: a.tasks_failed,
            })
            .collect()
    }

    pub fn get_agent(&self, role: AgentRole) -> Option<&Agent> {
        self.agents.get(&role)
    }

    pub fn active_count(&self) -> u8 {
        self.agents
            .values()
            .filter(|a| a.status == AgentStatus::Active)
            .count() as u8
    }

    pub fn idle_count(&self) -> u8 {
        self.agents
            .values()
            .filter(|a| a.status == AgentStatus::Idle)
            .count() as u8
    }

    pub fn create_goal(&mut self, description: String) -> &Goal {
        let id = self.next_goal_id;
        self.next_goal_id += 1;
        self.goals.push(Goal::new(id, description));
        self.goals.last().unwrap()
    }

    pub fn list_goals(&self) -> &[Goal] {
        &self.goals
    }

    pub fn get_goal(&self, id: u64) -> Option<&Goal> {
        self.goals.iter().find(|g| g.id == id)
    }

    /// Execute a goal. Marks involved agents as active, then idle.
    pub async fn execute_goal(
        &mut self,
        goal_id: u64,
        llm: &LlmRouter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Mark CEO active
        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Active;
        }

        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        let result = executor::execute_goal(llm, goal).await;

        // Update agent stats based on subtask results
        if let Some(goal) = self.goals.iter().find(|g| g.id == goal_id) {
            for subtask in &goal.subtasks {
                if let Some(agent) = self.agents.get_mut(&subtask.assigned_to) {
                    match subtask.status {
                        GoalStatus::Completed => agent.tasks_completed += 1,
                        GoalStatus::Failed => agent.tasks_failed += 1,
                        _ => {}
                    }
                    agent.status = AgentStatus::Idle;
                }
            }
        }

        // Mark CEO idle
        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Idle;
            match &result {
                Ok(_) => ceo.tasks_completed += 1,
                Err(_) => ceo.tasks_failed += 1,
            }
        }

        result
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}
