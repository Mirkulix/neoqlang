pub mod routes;

use axum::{
    routing::{get, post},
    Router,
};
use qo_agents::AgentRegistry;
use qo_consciousness::{ConsciousnessState, ConsciousnessStream};
use qo_evolution::{PatternDetector, ProposalEngine, QuantumState};
use qo_llm::LlmRouter;
use qo_memory::{ObsidianBridge, Store};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

pub struct AppState {
    pub llm: LlmRouter,
    pub store: Store,
    pub consciousness: Mutex<ConsciousnessState>,
    pub stream: ConsciousnessStream,
    pub obsidian: ObsidianBridge,
    pub agents: Mutex<AgentRegistry>,
    pub patterns: Mutex<PatternDetector>,
    pub proposals: Mutex<ProposalEngine>,
    pub quantum: Mutex<QuantumState>,
}

pub struct QoConfig {
    pub port: u16,
    pub groq_api_key: Option<String>,
    /// (api_key, base_url, model) for a custom cloud LLM
    pub cloud_config: Option<(String, String, String)>,
    pub data_dir: std::path::PathBuf,
    pub obsidian_vault: std::path::PathBuf,
    pub static_dir: Option<std::path::PathBuf>,
}

impl Default for QoConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            groq_api_key: None,
            cloud_config: None,
            data_dir: std::path::PathBuf::from("data"),
            obsidian_vault: std::path::PathBuf::from("vault"),
            static_dir: None,
        }
    }
}

pub fn build_app(
    config: QoConfig,
) -> Result<(Router, Arc<AppState>), Box<dyn std::error::Error + Send + Sync>> {
    let db_path = config.data_dir.join("qo.redb");
    // Ensure the data directory exists
    std::fs::create_dir_all(&config.data_dir)?;

    let store = Store::open(&db_path)?;
    let llm = LlmRouter::new(config.groq_api_key, config.cloud_config);
    let obsidian = ObsidianBridge::new(config.obsidian_vault);
    let stream = ConsciousnessStream::new(64);
    let consciousness = Mutex::new(ConsciousnessState::default());

    let agents = Mutex::new(AgentRegistry::new());
    let patterns = Mutex::new(PatternDetector::new());
    let proposals = Mutex::new(ProposalEngine::new());
    let quantum = Mutex::new(QuantumState::new(vec![
        "Direkte Ausführung".into(),
        "Dekomposition + Delegation".into(),
        "Recherche zuerst".into(),
        "Kreative Lösung".into(),
    ]));

    let state = Arc::new(AppState {
        llm,
        store,
        consciousness,
        stream,
        obsidian,
        agents,
        patterns,
        proposals,
        quantum,
    });

    let api_router = Router::new()
        .route("/api/health", get(routes::health::health))
        .route("/api/chat", post(routes::chat::chat))
        .route("/api/chat/history", get(routes::chat::chat_history))
        .route(
            "/api/consciousness/stream",
            get(routes::consciousness::stream),
        )
        .route(
            "/api/consciousness/state",
            get(routes::consciousness::current_state),
        )
        .route("/api/agents", get(routes::agents::list_agents))
        .route("/api/agents/{role}", get(routes::agents::get_agent))
        .route("/api/goals", get(routes::goals::list_goals))
        .route("/api/goals", post(routes::goals::create_goal))
        .route("/api/goals/{id}", get(routes::goals::get_goal))
        .route("/api/evolution/state", get(routes::evolution::quantum_state))
        .route("/api/evolution/patterns", get(routes::evolution::list_patterns))
        .route("/api/evolution/proposals", get(routes::evolution::list_proposals))
        .route("/api/evolution/proposals/{id}/approve", post(routes::evolution::approve_proposal))
        .route("/api/evolution/proposals/{id}/reject", post(routes::evolution::reject_proposal))
        .route("/api/evolution/analyze", post(routes::evolution::analyze))
        .route("/api/history", get(routes::history::get_history))
        .route("/api/goals/{id}/graph", get(routes::goals::get_goal_graph))
        .route("/api/providers", get(routes::providers::list_providers))
        .route("/api/providers/costs", get(routes::providers::cost_summary))
        .with_state(state.clone());

    let router = if let Some(static_dir) = config.static_dir {
        api_router.fallback_service(ServeDir::new(static_dir))
    } else {
        api_router
    };

    let router = router.layer(CorsLayer::permissive());

    Ok((router, state))
}
