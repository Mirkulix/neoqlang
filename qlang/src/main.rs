use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Load env vars from ~/.openclaw/.env if it exists
    let env_path = dirs_home().join(".openclaw/.env");
    if env_path.exists() {
        for line in std::fs::read_to_string(&env_path)?.lines() {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');
                if !key.is_empty() && !key.starts_with('#') {
                    std::env::set_var(key, value);
                }
            }
        }
    }

    let cloud_config = {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .ok();
        let base_url = std::env::var("CLOUD_BASE_URL").ok();
        let model = std::env::var("CLOUD_MODEL").ok();
        match (api_key, base_url, model) {
            (Some(k), Some(u), Some(m)) => Some((k, u, m)),
            _ => None,
        }
    };

    let config = qo_server::QoConfig {
        port: std::env::var("QO_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(4646),
        groq_api_key: std::env::var("GROQ_API_KEY").ok(),
        cloud_config,
        data_dir: PathBuf::from("data"),
        obsidian_vault: dirs_home().join("Dokumente/Obsidian Vault/Orbit"),
        static_dir: Some(PathBuf::from("frontend/dist")),
    };

    let port = config.port;
    let (app, state) = qo_server::build_app(config).map_err(|e| format!("{e}"))?;

    // Start heartbeat tick
    let cs_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            let mut cs = cs_state.consciousness.lock().await;
            cs.tick();
            cs_state.stream.publish(cs.clone());
        }
    });

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("QO starting on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
