use axum::{
    extract::{Path, State},
    Json,
};
use qo_agents::{ExecutionGraph, Goal};
use qo_evolution::SystemStats;
use qo_memory::graph_builders;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

fn serialize_goal(goal: &Goal) -> Option<String> {
    serde_json::to_string(goal).ok()
}

#[derive(Deserialize)]
pub struct CreateGoalRequest {
    pub description: String,
}

#[derive(Serialize)]
pub struct CreateGoalResponse {
    pub goal: Goal,
    pub message: String,
}

pub async fn list_goals(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Goal>>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let goals = registry.list_goals().to_vec();
    Ok(Json(goals))
}

pub async fn create_goal(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateGoalRequest>,
) -> Result<Json<CreateGoalResponse>, (axum::http::StatusCode, String)> {
    // Create the goal and get its id
    let goal_id = {
        let mut registry = state.agents.lock().await;
        let goal = registry.create_goal(req.description.clone());
        goal.id
    };

    // Retrieve the newly created goal for the response
    let goal = {
        let registry = state.agents.lock().await;
        registry
            .get_goal(goal_id)
            .cloned()
            .ok_or_else(|| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Goal not found after creation".to_string()))?
    };

    // Log action history
    if let Err(e) = state.store.log_action("goal_created", &req.description, "") {
        tracing::warn!("failed to log goal_created action: {e}");
    }

    // Persist goal to store
    if let Some(json) = serialize_goal(&goal) {
        if let Err(e) = state.store.save_goal(goal.id, &json) {
            tracing::warn!("failed to persist goal {}: {e}", goal.id);
        }
    }

    // Emit activity: goal created
    state.stream.publish_activity(
        format!("Neues Ziel erstellt: {}", req.description),
        None,
        "info",
    );

    // Spawn background task to execute the goal
    let state_clone = state.clone();
    let description = req.description.clone();
    tokio::spawn(async move {
        execute_goal_background(state_clone, goal_id, description).await;
    });

    Ok(Json(CreateGoalResponse {
        goal,
        message: "Ziel erstellt und wird im Hintergrund ausgeführt.".to_string(),
    }))
}

pub async fn execute_goal_background(
    state: Arc<AppState>,
    goal_id: u64,
    description: String,
) {
    let goal_start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    state.stream.publish_activity(
        "CEO analysiert Ziel...",
        Some("CEO".to_string()),
        "progress",
    );

    // We need to execute the goal step by step with activity events.
    // First, do the decomposition phase.
    let result = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_decompose(goal_id, &*state.llm).await
    };

    match result {
        Err(e) => {
            state.stream.publish_activity(
                format!("CEO ✗ Fehler bei Dekomposition: {}", e),
                Some("CEO".to_string()),
                "error",
            );
            return;
        }
        Ok(subtask_count) => {
            state.stream.publish_activity(
                format!("CEO hat {} Teilaufgaben erstellt", subtask_count),
                Some("CEO".to_string()),
                "success",
            );
        }
    }

    // Execute all subtasks in parallel — releases the lock before spawning
    state.stream.publish_activity(
        "Agenten arbeiten parallel an Teilaufgaben...".to_string(),
        None,
        "progress",
    );

    let parallel_outcomes = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_subtasks_parallel(goal_id, state.llm.clone()).await
    };

    // Publish per-subtask activity events and build results for graph
    let mut subtask_results: Vec<(String, String, bool, u64)> = Vec::new();
    for (_, agent_name, task_desc, succeeded, duration_ms) in &parallel_outcomes {
        if *succeeded {
            state.stream.publish_activity(
                format!("{} \u{2713} Aufgabe erledigt", agent_name),
                Some(agent_name.clone()),
                "success",
            );
        } else {
            state.stream.publish_activity(
                format!("{} \u{2717} Teilaufgabe fehlgeschlagen", agent_name),
                Some(agent_name.clone()),
                "error",
            );
        }
        subtask_results.push((agent_name.clone(), task_desc.clone(), *succeeded, *duration_ms));
    }

    // CEO summary
    state.stream.publish_activity(
        "CEO fasst Ergebnisse zusammen...",
        Some("CEO".to_string()),
        "progress",
    );

    let summary_result = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_summarize(goal_id, &*state.llm).await
    };

    match summary_result {
        Ok(summary) => {
            let short = if summary.len() > 80 {
                format!("{}...", &summary[..80])
            } else {
                summary.clone()
            };
            state.stream.publish_activity(
                format!("Ziel abgeschlossen: {}", short),
                Some("CEO".to_string()),
                "success",
            );
            if let Err(e) = state.store.log_action("goal_completed", &description, &short) {
                tracing::warn!("failed to log goal_completed action: {e}");
            }
        }
        Err(e) => {
            state.stream.publish_activity(
                format!("CEO ✗ Fehler bei Zusammenfassung: {}", e),
                Some("CEO".to_string()),
                "error",
            );
            tracing::warn!("Goal {} summary failed: {}", goal_id, e);
            if let Err(log_err) = state.store.log_action("goal_failed", &description, &e.to_string()) {
                tracing::warn!("failed to log goal_failed action: {log_err}");
            }
        }
    }

    // Update agent stats
    {
        let mut registry = state.agents.lock().await;
        registry.finalize_goal(goal_id);
    }

    // Mini-evolution: pattern analysis after goal completion
    {
        let (stats, _cs_energy) = {
            let agents = state.agents.lock().await;
            let cs = state.consciousness.lock().await;
            let agent_list = agents.list_agents();
            let active = agent_list.iter().filter(|a| a.status == qo_agents::AgentStatus::Active).count() as u8;
            let idle = agent_list.iter().filter(|a| a.status == qo_agents::AgentStatus::Idle).count() as u8;
            let completed: u32 = agent_list.iter().map(|a| a.tasks_completed).sum();
            let failed: u32 = agent_list.iter().map(|a| a.tasks_failed).sum();
            (SystemStats {
                total_tasks: completed + failed,
                tasks_completed: completed,
                tasks_failed: failed,
                agents_active: active,
                agents_idle: idle,
                avg_energy: cs.energy,
                completed_streak: 0,
            }, cs.energy)
        };
        let new_patterns = {
            let mut patterns = state.patterns.lock().await;
            patterns.analyze(&stats)
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        if !new_patterns.is_empty() {
            tracing::info!("Post-goal mini-evolution: {} new patterns", new_patterns.len());
        }
    }

    // Persist finalized goal and updated agent stats
    {
        let registry = state.agents.lock().await;
        if let Some(goal) = registry.get_goal(goal_id) {
            if let Some(json) = serialize_goal(goal) {
                if let Err(e) = state.store.save_goal(goal_id, &json) {
                    tracing::warn!("failed to persist finalized goal {}: {e}", goal_id);
                }
            }
        }
        for agent in registry.list_agents() {
            if let Ok(json) = serde_json::to_string(&agent) {
                let role_str = format!("{:?}", agent.role);
                if let Err(e) = state.store.save_agent_stats(&role_str, &json) {
                    tracing::warn!("failed to persist agent stats for {:?}: {e}", agent.role);
                }
            }
        }
    }

    // Build and store QLANG graph for this goal execution
    {
        let total_duration_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - goal_start_ms;
        let graph = graph_builders::build_goal_graph(&description, &subtask_results, total_duration_ms);
        if let Err(e) = state.graph_store.store(&graph) {
            tracing::warn!("failed to store goal graph: {e}");
        }
    }

    state.stream.publish_activity(
        format!("Ziel #{} vollständig abgeschlossen", goal_id),
        None,
        "info",
    );

    // Publish updated description for activity
    let _ = description;
}

pub async fn get_goal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<Goal>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let goal = registry.get_goal(id).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Goal {} not found", id),
        )
    })?;
    Ok(Json(goal.clone()))
}

pub async fn get_goal_graph(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ExecutionGraph>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let goal = registry.get_goal(id).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Goal {} not found", id),
        )
    })?;
    let graph = goal.execution_graph.clone().ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            "No execution graph available yet".to_string(),
        )
    })?;
    Ok(Json(graph))
}
