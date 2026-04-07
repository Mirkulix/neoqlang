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

    // Spawn background task to execute the goal
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut registry = state_clone.agents.lock().await;
        if let Err(e) = registry.execute_goal(goal_id, &state_clone.llm).await {
            tracing::warn!("Goal {} execution failed: {}", goal_id, e);
        }
    });

    Ok(Json(CreateGoalResponse {
        goal,
        message: "Ziel erstellt und wird im Hintergrund ausgeführt.".to_string(),
    }))
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
