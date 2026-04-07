use axum::{
    extract::{Path, State},
    Json,
};
use qo_agents::Goal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

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
    state.stream.publish_activity(
        "CEO analysiert Ziel...",
        Some("CEO".to_string()),
        "progress",
    );

    // We need to execute the goal step by step with activity events.
    // First, do the decomposition phase.
    let result = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_decompose(goal_id, &state.llm).await
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

    // Get subtask info for activity events
    let subtasks: Vec<(String, String)> = {
        let registry = state.agents.lock().await;
        if let Some(goal) = registry.get_goal(goal_id) {
            goal.subtasks.iter().map(|s| (s.assigned_to.name().to_string(), s.description.clone())).collect()
        } else {
            vec![]
        }
    };

    // Execute subtasks one by one with activity events
    for (i, (agent_name, task_desc)) in subtasks.iter().enumerate() {
        state.stream.publish_activity(
            format!("{} arbeitet an: {}", agent_name, task_desc),
            Some(agent_name.clone()),
            "progress",
        );

        let exec_result = {
            let mut registry = state.agents.lock().await;
            registry.execute_goal_subtask(goal_id, i, &state.llm).await
        };

        match exec_result {
            Ok(_) => {
                state.stream.publish_activity(
                    format!("{} \u{2713} Aufgabe erledigt", agent_name),
                    Some(agent_name.clone()),
                    "success",
                );
            }
            Err(e) => {
                state.stream.publish_activity(
                    format!("{} \u{2717} Fehler: {}", agent_name, e),
                    Some(agent_name.clone()),
                    "error",
                );
            }
        }
    }

    // CEO summary
    state.stream.publish_activity(
        "CEO fasst Ergebnisse zusammen...",
        Some("CEO".to_string()),
        "progress",
    );

    let summary_result = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_summarize(goal_id, &state.llm).await
    };

    match summary_result {
        Ok(summary) => {
            let short = if summary.len() > 80 {
                format!("{}...", &summary[..80])
            } else {
                summary
            };
            state.stream.publish_activity(
                format!("Ziel abgeschlossen: {}", short),
                Some("CEO".to_string()),
                "success",
            );
        }
        Err(e) => {
            state.stream.publish_activity(
                format!("CEO ✗ Fehler bei Zusammenfassung: {}", e),
                Some("CEO".to_string()),
                "error",
            );
            tracing::warn!("Goal {} summary failed: {}", goal_id, e);
        }
    }

    // Update agent stats
    {
        let mut registry = state.agents.lock().await;
        registry.finalize_goal(goal_id);
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
