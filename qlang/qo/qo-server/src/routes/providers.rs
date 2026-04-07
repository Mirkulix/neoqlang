use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProvidersResponse {
    pub providers: Vec<qo_llm::ProviderStats>,
    pub total_cost_usd: f64,
    pub total_requests: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CostSummaryResponse {
    pub total_cost_usd: f64,
    pub cloud_cost_usd: f64,
    pub groq_cost_usd: f64,
    pub local_cost_usd: f64,
    pub total_requests: u64,
    pub savings_vs_all_cloud_usd: f64,
}

pub async fn list_providers(State(state): State<Arc<AppState>>) -> Json<ProvidersResponse> {
    let providers = state.llm.provider_stats();
    let total_cost = state.llm.total_cost();
    let total_requests: u64 = providers.iter().map(|p| p.requests).sum();

    Json(ProvidersResponse {
        providers,
        total_cost_usd: total_cost,
        total_requests,
    })
}

pub async fn cost_summary(State(state): State<Arc<AppState>>) -> Json<CostSummaryResponse> {
    let providers = state.llm.provider_stats();

    let groq_cost = providers
        .iter()
        .find(|p| p.name == "Groq")
        .map(|p| p.cost_usd)
        .unwrap_or(0.0);
    let cloud_cost = providers
        .iter()
        .find(|p| p.name == "Cloud")
        .map(|p| p.cost_usd)
        .unwrap_or(0.0);
    let local_cost = 0.0f64;

    let total_cost = groq_cost + cloud_cost + local_cost;
    let total_requests: u64 = providers.iter().map(|p| p.requests).sum();

    // Savings: what would have been spent if all requests used Claude at $0.01/1K tokens avg
    // Estimate 300 tokens per request on average
    let total_tokens_estimate: u64 = providers.iter().map(|p| p.total_tokens_estimate).sum();
    let hypothetical_cloud_cost = (total_tokens_estimate as f64 / 1000.0) * 0.01;
    let savings = hypothetical_cloud_cost - total_cost;

    Json(CostSummaryResponse {
        total_cost_usd: total_cost,
        cloud_cost_usd: cloud_cost,
        groq_cost_usd: groq_cost,
        local_cost_usd: local_cost,
        total_requests,
        savings_vs_all_cloud_usd: savings.max(0.0),
    })
}
