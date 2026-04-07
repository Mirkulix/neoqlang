use axum::{extract::State, Json};
use qo_consciousness::StateEvent;
use qo_memory::graph_builders;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

fn looks_like_goal(msg: &str) -> bool {
    let prefixes = [
        "ziel:", "goal:", "mach:", "recherchiere", "plane", "baue",
        "erstelle", "analysiere",
    ];
    let lower = msg.to_lowercase();
    prefixes.iter().any(|p| lower.starts_with(p))
}

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
    // Detect goal-like messages and route them to the agent system
    let goal_prefix = if looks_like_goal(&req.message) {
        let goal_id = {
            let mut registry = state.agents.lock().await;
            let goal = registry.create_goal(req.message.clone());
            goal.id
        };

        state.stream.publish_activity(
            format!("Chat-Ziel #{} erstellt und wird bearbeitet", goal_id),
            None,
            "info",
        );

        // Spawn background execution
        let state_clone = state.clone();
        let description = req.message.clone();
        tokio::spawn(async move {
            crate::routes::goals::execute_goal_background(state_clone, goal_id, description).await;
        });

        Some(format!(
            "[Ziel #{} erstellt und wird von den Agenten bearbeitet.]\n\n",
            goal_id
        ))
    } else {
        None
    };

    let messages = vec![
        ("system".to_string(), SYSTEM_PROMPT.to_string()),
        ("user".to_string(), req.message.clone()),
    ];

    let llm_start = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // 1. Load user-configured providers from redb (enabled, sorted by tier ascending)
    let mut configured: Vec<qo_llm::ProviderConfig> = state
        .store
        .list_providers()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(_, json)| serde_json::from_str::<qo_llm::ProviderConfig>(&json).ok())
        .filter(|p| p.enabled)
        .collect();
    configured.sort_by_key(|p| p.tier);

    // 2. Try each configured provider in tier order; first success wins
    let mut provider_used = String::new();
    let mut llm_response_opt: Option<String> = None;

    for provider in &configured {
        let base_url = provider.base_url.clone().unwrap_or_else(|| {
            "https://api.openai.com/v1".to_string()
        });
        match state
            .llm
            .chat_with_provider(&base_url, &provider.api_key, &provider.model, messages.clone())
            .await
        {
            Ok(resp) => {
                provider_used = provider.id.clone();
                llm_response_opt = Some(resp);
                break;
            }
            Err(e) => {
                tracing::warn!(provider = %provider.id, "configured provider failed: {e}");
            }
        }
    }

    // 3. Fall back to env-based router (Groq / Cloud)
    let llm_response = if let Some(resp) = llm_response_opt {
        resp
    } else {
        provider_used = "groq".to_string();
        state
            .llm
            .chat(messages)
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };

    let llm_duration_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
        - llm_start;

    let response = match goal_prefix {
        Some(prefix) => format!("{}{}", prefix, llm_response),
        None => llm_response,
    };

    // Store in redb
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let entry = serde_json::json!({
        "id": id,
        "user": req.message,
        "assistant": response,
        "timestamp": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    });
    let entry_str = entry.to_string();

    if let Err(e) = state.store.store_chat(id, &entry_str) {
        tracing::warn!("failed to store chat in redb: {e}");
    }

    // Build and store QLANG graph for this chat interaction
    {
        let graph = graph_builders::build_chat_graph(
            &req.message,
            &response,
            &provider_used,
            llm_duration_ms,
        );
        if let Err(e) = state.graph_store.store(&graph) {
            tracing::warn!("failed to store chat graph: {e}");
        }
    }

    // Log action history
    let response_short = if response.len() > 100 {
        format!("{}...", &response[..100])
    } else {
        response.clone()
    };
    if let Err(e) = state.store.log_action("chat", &req.message, &response_short) {
        tracing::warn!("failed to log chat action: {e}");
    }

    // Update consciousness and log to Obsidian
    {
        let mut cs = state.consciousness.lock().await;
        cs.process_event(&StateEvent::ChatReceived);
        let mood_str = format!("{:?}", cs.mood);
        let energy = cs.energy;
        let heartbeat = cs.heartbeat;
        state.stream.publish(cs.clone());
        drop(cs);

        if let Err(e) = state
            .obsidian
            .log_consciousness_event(&mood_str, energy, heartbeat, "ChatReceived")
            .await
        {
            tracing::warn!("failed to log consciousness event to Obsidian: {e}");
        }
    }

    Ok(Json(ChatResponse {
        response,
        tier: provider_used,
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
