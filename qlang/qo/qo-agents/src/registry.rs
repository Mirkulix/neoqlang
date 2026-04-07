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

    /// Step 1: CEO decomposes the goal into subtasks. Returns subtask count.
    pub async fn execute_goal_decompose(
        &mut self,
        goal_id: u64,
        llm: &LlmRouter,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Active;
        }

        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        executor::decompose_goal(llm, goal).await?;
        let count = goal.subtasks.len();
        Ok(count)
    }

    /// Step 2: Execute a single subtask by index.
    pub async fn execute_goal_subtask(
        &mut self,
        goal_id: u64,
        subtask_index: usize,
        llm: &LlmRouter,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Mark the agent for this subtask as active
        let agent_role = {
            let goal = self
                .goals
                .iter()
                .find(|g| g.id == goal_id)
                .ok_or("Goal not found")?;
            goal.subtasks
                .get(subtask_index)
                .ok_or("Subtask not found")?
                .assigned_to
        };

        if let Some(agent) = self.agents.get_mut(&agent_role) {
            agent.status = AgentStatus::Active;
        }

        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        let result = executor::execute_subtask(llm, goal, subtask_index).await;

        if let Some(agent) = self.agents.get_mut(&agent_role) {
            agent.status = AgentStatus::Idle;
            match &result {
                Ok(_) => agent.tasks_completed += 1,
                Err(_) => agent.tasks_failed += 1,
            }
        }

        result
    }

    /// Step 3: CEO summarizes all subtask results. Returns summary string.
    pub async fn execute_goal_summarize(
        &mut self,
        goal_id: u64,
        llm: &LlmRouter,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        executor::summarize_goal(llm, goal).await
    }

    /// Finalize: update CEO stats and mark goal complete.
    pub fn finalize_goal(&mut self, goal_id: u64) {
        let succeeded = self
            .goals
            .iter()
            .find(|g| g.id == goal_id)
            .map(|g| g.result.is_some())
            .unwrap_or(false);

        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Idle;
            if succeeded {
                ceo.tasks_completed += 1;
            } else {
                ceo.tasks_failed += 1;
            }
        }
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}
