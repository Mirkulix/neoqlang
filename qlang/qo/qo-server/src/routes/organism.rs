//! Organism API — interact with the QLANG swarm organism.

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::AppState;

#[derive(Deserialize)]
pub struct OrganismInput {
    pub message: String,
}

#[derive(Serialize)]
pub struct OrganismOutput {
    pub text: String,
    pub specialist: String,
    pub confidence: f32,
    pub reasoning: Vec<String>,
    pub memory_stored: bool,
    pub specialists: Vec<SpecialistInfo>,
    pub total_interactions: usize,
    pub generation: u32,
    pub memory_items: usize,
}

#[derive(Serialize)]
pub struct SpecialistInfo {
    pub name: String,
    pub invocations: u64,
    pub success_rate: f32,
}

use once_cell::sync::Lazy;
static ORGANISM: Lazy<Mutex<qlang_runtime::organism::Organism>> = Lazy::new(|| {
    Mutex::new(qlang_runtime::organism::Organism::new(1000))
});

/// POST /api/organism/chat — send message to organism
pub async fn chat(Json(input): Json<OrganismInput>) -> Json<OrganismOutput> {
    let mut org = ORGANISM.lock().await;
    let resp = org.process(&input.message);

    let specialists: Vec<SpecialistInfo> = org.specialists.iter().map(|s| SpecialistInfo {
        name: s.name.clone(),
        invocations: s.invocations,
        success_rate: s.success_rate,
    }).collect();

    Json(OrganismOutput {
        text: resp.text,
        specialist: resp.specialist,
        confidence: resp.confidence,
        reasoning: resp.reasoning,
        memory_stored: resp.memory_stored,
        specialists,
        total_interactions: org.total_interactions(),
        generation: org.generation,
        memory_items: org.shared_memory.items.len(),
    })
}

/// POST /api/organism/evolve — trigger evolution
pub async fn evolve() -> Json<serde_json::Value> {
    let mut org = ORGANISM.lock().await;
    org.evolve();
    Json(serde_json::json!({
        "generation": org.generation,
        "specialists": org.specialist_count(),
        "interactions": org.total_interactions(),
    }))
}

/// GET /api/organism/status — get organism status
pub async fn status() -> Json<serde_json::Value> {
    let org = ORGANISM.lock().await;
    let specialists: Vec<serde_json::Value> = org.specialists.iter().map(|s| {
        serde_json::json!({
            "name": s.name,
            "invocations": s.invocations,
            "success_rate": s.success_rate,
        })
    }).collect();

    Json(serde_json::json!({
        "generation": org.generation,
        "specialists": specialists,
        "total_interactions": org.total_interactions(),
        "memory_items": org.shared_memory.items.len(),
    }))
}
