use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub tier: String,
}

const SYSTEM_PROMPT: &str =
    "Du bist QO, ein persönlicher KI-Companion. Antworte auf Deutsch.";

pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (axum::http::StatusCode, String)> {
    let messages = vec![
        ("system".to_string(), SYSTEM_PROMPT.to_string()),
        ("user".to_string(), req.message.clone()),
    ];

    let response = state
        .llm
        .chat(messages)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Store in redb
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let entry = serde_json::json!({
        "id": id,
        "user": req.message,
        "assistant": response,
    });
    let entry_str = entry.to_string();

    if let Err(e) = state.store.store_chat(id, &entry_str) {
        tracing::warn!("failed to store chat in redb: {e}");
    }

    // Update consciousness
    {
        let mut cs = state.consciousness.lock().await;
        cs.drain_energy(5.0);
        cs.task_completed();
        state.stream.publish(cs.clone());
    }

    Ok(Json(ChatResponse {
        response,
        tier: "groq".to_string(),
    }))
}

pub async fn chat_history(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Value>>, (axum::http::StatusCode, String)> {
    let history = state
        .store
        .chat_history(50)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<Value> = history
        .into_iter()
        .filter_map(|(_, json_str)| serde_json::from_str(&json_str).ok())
        .collect();

    Ok(Json(items))
}
