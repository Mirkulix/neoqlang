//! Evolution Daemon HTTP API — drives the EvolutionDashboard frontend.
//!
//! Endpoints:
//!   POST /api/evolution/start        — start daemon with optional config
//!   POST /api/evolution/stop         — stop the daemon
//!   GET  /api/evolution/status       — current status + last report
//!   GET  /api/evolution/history      — all generation reports (most recent 500)
//!   GET  /api/evolution/specialists  — active + retired specialists with fitness
//!   GET  /api/evolution/lineage/:id  — lineage tree (ancestors + descendants)
//!   GET  /api/evolution/stream       — SSE stream of new GenerationReports
//!
//! State: a shared `Arc<Mutex<Option<SharedDaemon>>>` inside `AppState` so the
//! daemon can be (re)started from the UI.

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use qlang_runtime::evolution::daemon::{
    DaemonConfig, DaemonStatus, EvolutionDaemon, GenerationReport, SharedDaemon, SpecialistInfo,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

use crate::AppState;

// ---------------------------------------------------------------------------
// Request / Response payloads
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub struct StartRequest {
    pub interval_secs: Option<u64>,
    pub population_size: Option<usize>,
    pub mutation_rate: Option<f32>,
    pub retire_fraction: Option<f32>,
    pub max_age: Option<u32>,
    pub seed: Option<u64>,
}

impl StartRequest {
    fn into_config(self) -> DaemonConfig {
        let defaults = DaemonConfig::default();
        DaemonConfig {
            interval_secs: self.interval_secs.unwrap_or(defaults.interval_secs).max(1),
            population_size: self
                .population_size
                .unwrap_or(defaults.population_size)
                .clamp(4, 500),
            mutation_rate: self
                .mutation_rate
                .unwrap_or(defaults.mutation_rate)
                .clamp(0.0, 1.0),
            retire_fraction: self
                .retire_fraction
                .unwrap_or(defaults.retire_fraction)
                .clamp(0.0, 0.9),
            max_age: self.max_age.unwrap_or(defaults.max_age).max(1),
            seed: self.seed.unwrap_or(defaults.seed),
        }
    }
}

#[derive(Serialize)]
pub struct SimpleResponse {
    pub ok: bool,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn get_or_create(state: &AppState) -> SharedDaemon {
    let mut slot = state.evolution_daemon.lock().await;
    if slot.is_none() {
        *slot = Some(EvolutionDaemon::new());
    }
    slot.as_ref().unwrap().clone()
}

async fn current(state: &AppState) -> Option<SharedDaemon> {
    state.evolution_daemon.lock().await.clone()
}

// ---------------------------------------------------------------------------
// POST /api/evolution/start
// ---------------------------------------------------------------------------

pub async fn start(
    State(state): State<Arc<AppState>>,
    body: Option<Json<StartRequest>>,
) -> Json<SimpleResponse> {
    let cfg = body.map(|Json(r)| r).unwrap_or_default().into_config();
    let daemon = get_or_create(&state).await;
    match daemon.start(cfg) {
        Ok(()) => Json(SimpleResponse {
            ok: true,
            message: "evolution daemon started".into(),
        }),
        Err(e) => Json(SimpleResponse {
            ok: false,
            message: e,
        }),
    }
}

// ---------------------------------------------------------------------------
// POST /api/evolution/stop
// ---------------------------------------------------------------------------

pub async fn stop(State(state): State<Arc<AppState>>) -> Json<SimpleResponse> {
    if let Some(d) = current(&state).await {
        d.stop();
        Json(SimpleResponse {
            ok: true,
            message: "stop signal sent".into(),
        })
    } else {
        Json(SimpleResponse {
            ok: false,
            message: "daemon not initialized".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// GET /api/evolution/status
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct StatusResponse {
    pub initialized: bool,
    #[serde(flatten)]
    pub status: Option<DaemonStatus>,
}

pub async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    if let Some(d) = current(&state).await {
        Json(StatusResponse {
            initialized: true,
            status: Some(d.status()),
        })
    } else {
        Json(StatusResponse {
            initialized: false,
            status: None,
        })
    }
}

// ---------------------------------------------------------------------------
// GET /api/evolution/history
// ---------------------------------------------------------------------------

pub async fn history(State(state): State<Arc<AppState>>) -> Json<Vec<GenerationReport>> {
    match current(&state).await {
        Some(d) => Json(d.history()),
        None => Json(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// GET /api/evolution/specialists
// ---------------------------------------------------------------------------

pub async fn specialists(State(state): State<Arc<AppState>>) -> Json<Vec<SpecialistInfo>> {
    match current(&state).await {
        Some(d) => Json(d.specialists()),
        None => Json(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// GET /api/evolution/lineage/:id
// ---------------------------------------------------------------------------

pub async fn lineage(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<Vec<SpecialistInfo>> {
    match current(&state).await {
        Some(d) => Json(d.lineage(&id)),
        None => Json(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// GET /api/evolution/stream — SSE of new generation reports
// ---------------------------------------------------------------------------

pub async fn stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let daemon = get_or_create(&state).await;
    let mut rx = daemon.subscribe();

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(report) => {
                    let data = serde_json::to_string(&report).unwrap_or_default();
                    yield Ok(Event::default().event("generation").data(data));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}
