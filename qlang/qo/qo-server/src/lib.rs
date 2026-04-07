pub mod auth;
pub mod routes;

use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};
use qo_agents::{AgentRegistry, AgentRole};
use qo_consciousness::{ConsciousnessState, ConsciousnessStream};
use qo_evolution::{Pattern, PatternDetector, Proposal, ProposalEngine, QuantumState};
use qo_llm::LlmRouter;
use qo_memory::{GraphStore, ObsidianBridge, Store};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

pub struct AppState {
    pub llm: Arc<LlmRouter>,
    pub store: Store,
    pub graph_store: GraphStore,
    pub consciousness: Mutex<ConsciousnessState>,
    pub stream: ConsciousnessStream,
    pub obsidian: ObsidianBridge,
    pub agents: Mutex<AgentRegistry>,
    pub patterns: Mutex<PatternDetector>,
    pub proposals: Mutex<ProposalEngine>,
    pub quantum: Mutex<QuantumState>,
    pub configured_providers: Mutex<Vec<qo_llm::ProviderConfig>>,
}

pub struct QoConfig {
    pub port: u16,
    pub groq_api_key: Option<String>,
    /// (api_key, base_url, model) for a custom cloud LLM
    pub cloud_config: Option<(String, String, String)>,
    pub data_dir: std::path::PathBuf,
    pub obsidian_vault: std::path::PathBuf,
    pub static_dir: Option<std::path::PathBuf>,
    /// Optional API token for bearer auth (reads QO_AUTH_TOKEN from env if None)
    pub auth_token: Option<String>,
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
            auth_token: None,
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
    let graph_store = GraphStore::new(store.db())?;
    let llm = Arc::new(LlmRouter::new(config.groq_api_key, config.cloud_config));
    let obsidian = ObsidianBridge::new(config.obsidian_vault);
    let stream = ConsciousnessStream::new(64);
    let consciousness = Mutex::new(ConsciousnessState::default());

    // Load persisted data BEFORE creating AppState (no async runtime yet)
    let mut agents_reg = AgentRegistry::new();
    let mut pattern_det = PatternDetector::new();
    let mut proposal_eng = ProposalEngine::new();
    let mut quantum_st = QuantumState::new(vec![
        "Direkte Ausführung".into(),
        "Dekomposition + Delegation".into(),
        "Recherche zuerst".into(),
        "Kreative Lösung".into(),
    ]);

    // Restore goals
    if let Ok(goals) = store.list_goals() {
        for (_, json) in goals {
            if let Ok(goal) = serde_json::from_str::<qo_agents::Goal>(&json) {
                agents_reg.restore_goal(goal);
            }
        }
    }

    // Restore agent stats
    if let Ok(agent_stats) = store.load_agent_stats() {
        for (role_str, json) in agent_stats {
            let role = match role_str.as_str() {
                "Ceo" => Some(AgentRole::Ceo),
                "Researcher" => Some(AgentRole::Researcher),
                "Developer" => Some(AgentRole::Developer),
                "Guardian" => Some(AgentRole::Guardian),
                "Strategist" => Some(AgentRole::Strategist),
                "Artisan" => Some(AgentRole::Artisan),
                _ => None,
            };
            if let Some(role) = role {
                #[derive(serde::Deserialize)]
                struct Stats { tasks_completed: u32, tasks_failed: u32 }
                if let Ok(stats) = serde_json::from_str::<Stats>(&json) {
                    agents_reg.restore_agent_stats(role, stats.tasks_completed, stats.tasks_failed);
                }
            }
        }
    }

    // Restore patterns
    if let Ok(data) = store.list_patterns() {
        for (_, json) in data {
            if let Ok(p) = serde_json::from_str::<Pattern>(&json) {
                pattern_det.restore_pattern(p);
            }
        }
    }

    // Restore proposals
    if let Ok(data) = store.list_proposals() {
        for (_, json) in data {
            if let Ok(p) = serde_json::from_str::<Proposal>(&json) {
                proposal_eng.restore_proposal(p);
            }
        }
    }

    // Restore quantum state
    if let Ok(Some(json)) = store.load_quantum_state() {
        if let Ok(qs) = serde_json::from_str::<QuantumState>(&json) {
            quantum_st = qs;
        }
    }

    // Load configured providers from redb so they are available for routing on startup
    let mut configured_providers = Vec::new();
    if let Ok(providers) = store.list_providers() {
        for (_, json) in providers {
            if let Ok(cfg) = serde_json::from_str::<qo_llm::ProviderConfig>(&json) {
                if cfg.enabled {
                    configured_providers.push(cfg);
                }
            }
        }
    }
    tracing::info!(
        "Loaded {} configured providers from store",
        configured_providers.len()
    );

    tracing::info!("Restored: {} goals, {} patterns, {} proposals, gen {}",
        agents_reg.list_goals().len(),
        pattern_det.all_patterns().len(),
        proposal_eng.all().len(),
        quantum_st.generation,
    );

    let state = Arc::new(AppState {
        llm,
        store,
        graph_store,
        consciousness,
        stream,
        obsidian,
        agents: Mutex::new(agents_reg),
        patterns: Mutex::new(pattern_det),
        proposals: Mutex::new(proposal_eng),
        quantum: Mutex::new(quantum_st),
        configured_providers: Mutex::new(configured_providers),
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
        .route("/api/graphs", get(routes::graphs::list_graphs))
        .route("/api/graphs/stats", get(routes::graphs::graph_stats))
        .route("/api/graphs/{id}", get(routes::graphs::get_graph))
        .route("/api/providers", get(routes::providers::list_providers))
        .route("/api/providers/costs", get(routes::providers::cost_summary))
        .route("/api/providers/templates", get(routes::providers::list_templates))
        .route("/api/providers/configured", get(routes::providers::list_configured))
        .route("/api/providers/add", post(routes::providers::add_provider))
        .route("/api/providers/test", post(routes::providers::test_provider))
        .route("/api/providers/{id}/toggle", put(routes::providers::toggle_provider))
        .route("/api/providers/{id}/edit", put(routes::providers::update_provider))
        .route("/api/providers/{id}", delete(routes::providers::delete_provider))
        .route("/api/simulation/run", post(routes::simulation::run_simulation))
        .route("/api/simulation/strategies", get(routes::simulation::list_strategies))
        .layer(middleware::from_fn(auth::auth_middleware))
        .with_state(state.clone());

    let router = if let Some(static_dir) = config.static_dir {
        api_router.fallback_service(ServeDir::new(static_dir))
    } else {
        api_router
    };

    let router = router.layer(CorsLayer::permissive());

    Ok((router, state))
}
