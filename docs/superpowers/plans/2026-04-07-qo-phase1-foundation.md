# QO Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** QO binary starts, serves React frontend, accepts chat via HTTP, routes to Groq, stores in HNSW+Obsidian, streams consciousness via WebSocket.

**Architecture:** Single Rust binary using Axum. Six new crates under `qlang/qo/`. Reuses existing `qlang-runtime` provider clients (Groq, Anthropic, etc.). React frontend copied from Orbit and adapted to new API.

**Tech Stack:** Rust (Axum, tokio, instant-distance, redb, tower-http), React 19 (Vite, TypeScript), existing qlang-* crates.

**Spec:** `docs/superpowers/specs/2026-04-07-qo-design.md`

---

## File Structure

```
qlang/
├── Cargo.toml                          (MODIFY — add qo workspace members)
├── qo/
│   ├── qo-server/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  (server startup, router assembly)
│   │       ├── routes/
│   │       │   ├── mod.rs
│   │       │   ├── chat.rs             (POST /api/chat, GET /api/chat/history)
│   │       │   ├── health.rs           (GET /api/health)
│   │       │   └── consciousness.rs    (GET /api/consciousness/stream SSE)
│   │       └── ws.rs                   (WebSocket upgrade for consciousness)
│   │
│   ├── qo-llm/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── router.rs              (complexity scoring → tier selection)
│   │       ├── groq.rs                (Groq API client, async)
│   │       └── cloud.rs              (Claude/DeepSeek fallback)
│   │
│   ├── qo-memory/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── obsidian.rs            (read/write Markdown to Obsidian Vault)
│   │       ├── hnsw.rs               (HNSW vector store via instant-distance)
│   │       ├── store.rs              (redb key-value persistent storage)
│   │       └── embeddings.rs         (text → vector, placeholder for Phase 4)
│   │
│   ├── qo-consciousness/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── state_machine.rs       (Mood, Energy, Heartbeat)
│   │       └── stream.rs             (broadcast channel for SSE/WS)
│   │
│   └── qo-values/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── scorer.rs              (5 values × f32 scoring)
│
├── frontend/                           (COPY from Orbit, adapt)
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx                    (simplified: Chat + Consciousness)
│       ├── ChatView.tsx               (new, talks to /api/chat)
│       └── ConsciousnessStream.tsx    (new, SSE from /api/consciousness/stream)
│
└── src/
    ├── main.rs                        (NEW — QO binary entry point)
    └── lib.rs                         (existing, keep)
```

---

### Task 1: Extend Cargo Workspace

**Files:**
- Modify: `qlang/Cargo.toml`

- [ ] **Step 1: Add qo crates to workspace members**

In `qlang/Cargo.toml`, add the new QO crates to the workspace members list and add workspace dependencies:

```toml
[workspace]
members = [
    "crates/qlang-core",
    "crates/qlang-compile",
    "crates/qlang-runtime",
    "crates/qlang-agent",
    "crates/qlang-python",
    "crates/qlang-sdk",
    "qo/qo-server",
    "qo/qo-llm",
    "qo/qo-memory",
    "qo/qo-consciousness",
    "qo/qo-values",
]

# Add to [workspace.dependencies]:
axum = { version = "0.8", features = ["ws"] }
tower-http = { version = "0.6", features = ["fs", "cors"] }
tokio = { version = "1.51.0", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
redb = "2"
instant-distance = "0.6"
tracing = "0.1"
tracing-subscriber = "0.3"
```

- [ ] **Step 2: Verify workspace compiles (empty crates)**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo check --workspace 2>&1 | tail -5`

Expected: Error because qo crates don't exist yet. That's fine — we create them next.

- [ ] **Step 3: Commit**

```bash
cd /home/mirkulix/neoqlang/qlang
git add Cargo.toml
git commit -m "feat(qo): add QO crates to workspace"
```

---

### Task 2: Create qo-values Crate (Leaf Dependency)

**Files:**
- Create: `qlang/qo/qo-values/Cargo.toml`
- Create: `qlang/qo/qo-values/src/lib.rs`
- Create: `qlang/qo/qo-values/src/scorer.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "qo-values"
version.workspace = true
edition.workspace = true
description = "QO values system — 5 core values scoring"

[dependencies]
serde = { workspace = true }
```

- [ ] **Step 2: Write scorer.rs**

```rust
use serde::{Deserialize, Serialize};

/// The 5 core values of QO.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Value {
    Achtsamkeit,
    Anerkennung,
    Aufmerksamkeit,
    Entwicklung,
    Sinn,
}

impl Value {
    pub const ALL: [Value; 5] = [
        Value::Achtsamkeit,
        Value::Anerkennung,
        Value::Aufmerksamkeit,
        Value::Entwicklung,
        Value::Sinn,
    ];
}

/// Scores for all 5 values, each in [0.0, 1.0].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueScores {
    pub achtsamkeit: f32,
    pub anerkennung: f32,
    pub aufmerksamkeit: f32,
    pub entwicklung: f32,
    pub sinn: f32,
}

impl Default for ValueScores {
    fn default() -> Self {
        Self {
            achtsamkeit: 0.5,
            anerkennung: 0.5,
            aufmerksamkeit: 0.5,
            entwicklung: 0.5,
            sinn: 0.5,
        }
    }
}

impl ValueScores {
    /// Get score for a specific value.
    pub fn get(&self, value: Value) -> f32 {
        match value {
            Value::Achtsamkeit => self.achtsamkeit,
            Value::Anerkennung => self.anerkennung,
            Value::Aufmerksamkeit => self.aufmerksamkeit,
            Value::Entwicklung => self.entwicklung,
            Value::Sinn => self.sinn,
        }
    }

    /// Set score for a specific value. Clamps to [0.0, 1.0].
    pub fn set(&mut self, value: Value, score: f32) {
        let clamped = score.clamp(0.0, 1.0);
        match value {
            Value::Achtsamkeit => self.achtsamkeit = clamped,
            Value::Anerkennung => self.anerkennung = clamped,
            Value::Aufmerksamkeit => self.aufmerksamkeit = clamped,
            Value::Entwicklung => self.entwicklung = clamped,
            Value::Sinn => self.sinn = clamped,
        }
    }

    /// Average across all values.
    pub fn average(&self) -> f32 {
        (self.achtsamkeit + self.anerkennung + self.aufmerksamkeit
            + self.entwicklung + self.sinn)
            / 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scores_are_half() {
        let s = ValueScores::default();
        for v in Value::ALL {
            assert!((s.get(v) - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn set_clamps_to_range() {
        let mut s = ValueScores::default();
        s.set(Value::Sinn, 1.5);
        assert!((s.sinn - 1.0).abs() < f32::EPSILON);
        s.set(Value::Sinn, -0.3);
        assert!(s.sinn.abs() < f32::EPSILON);
    }

    #[test]
    fn average_works() {
        let s = ValueScores {
            achtsamkeit: 1.0,
            anerkennung: 0.0,
            aufmerksamkeit: 0.5,
            entwicklung: 0.5,
            sinn: 1.0,
        };
        assert!((s.average() - 0.6).abs() < f32::EPSILON);
    }
}
```

- [ ] **Step 3: Write lib.rs**

```rust
pub mod scorer;
pub use scorer::{Value, ValueScores};
```

- [ ] **Step 4: Run tests**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo test -p qo-values`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add qo/qo-values/
git commit -m "feat(qo): add qo-values crate with 5-value scoring"
```

---

### Task 3: Create qo-consciousness Crate

**Files:**
- Create: `qlang/qo/qo-consciousness/Cargo.toml`
- Create: `qlang/qo/qo-consciousness/src/lib.rs`
- Create: `qlang/qo/qo-consciousness/src/state_machine.rs`
- Create: `qlang/qo/qo-consciousness/src/stream.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "qo-consciousness"
version.workspace = true
edition.workspace = true
description = "QO consciousness — state machine + broadcast stream"

[dependencies]
qo-values = { path = "../qo-values" }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }
```

- [ ] **Step 2: Write state_machine.rs**

```rust
use qo_values::ValueScores;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mood {
    Learning,
    Focused,
    Restless,
    Creating,
    Reflecting,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub mood: Mood,
    pub energy: f32,
    pub heartbeat: u64,
    pub agents_active: u8,
    pub agents_idle: u8,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub values: ValueScores,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            mood: Mood::Learning,
            energy: 100.0,
            heartbeat: 0,
            agents_active: 0,
            agents_idle: 6,
            tasks_completed: 0,
            tasks_failed: 0,
            values: ValueScores::default(),
        }
    }
}

impl ConsciousnessState {
    /// Advance heartbeat by one tick. Returns the new state.
    pub fn tick(&mut self) {
        self.heartbeat += 1;
        // Regenerate energy when idle
        if self.agents_active == 0 {
            self.energy = (self.energy + 0.1).min(100.0);
        }
    }

    /// Drain energy for an LLM call.
    pub fn drain_energy(&mut self, amount: f32) {
        self.energy = (self.energy - amount).max(0.0);
        if self.energy < 20.0 {
            self.mood = Mood::Restless;
        }
    }

    /// Record a completed task.
    pub fn task_completed(&mut self) {
        self.tasks_completed += 1;
        self.mood = Mood::Focused;
    }

    /// Record a failed task.
    pub fn task_failed(&mut self) {
        self.tasks_failed += 1;
        if self.tasks_failed > self.tasks_completed {
            self.mood = Mood::Restless;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state() {
        let s = ConsciousnessState::default();
        assert_eq!(s.mood, Mood::Learning);
        assert!((s.energy - 100.0).abs() < f32::EPSILON);
        assert_eq!(s.heartbeat, 0);
    }

    #[test]
    fn tick_increments_heartbeat() {
        let mut s = ConsciousnessState::default();
        s.tick();
        assert_eq!(s.heartbeat, 1);
        s.tick();
        assert_eq!(s.heartbeat, 2);
    }

    #[test]
    fn drain_energy_sets_restless() {
        let mut s = ConsciousnessState::default();
        s.drain_energy(85.0);
        assert!((s.energy - 15.0).abs() < f32::EPSILON);
        assert_eq!(s.mood, Mood::Restless);
    }

    #[test]
    fn energy_regens_on_idle_tick() {
        let mut s = ConsciousnessState::default();
        s.energy = 50.0;
        s.agents_active = 0;
        s.tick();
        assert!(s.energy > 50.0);
    }
}
```

- [ ] **Step 3: Write stream.rs**

```rust
use crate::state_machine::ConsciousnessState;
use tokio::sync::broadcast;

/// Event broadcast for consciousness updates.
#[derive(Debug, Clone)]
pub struct ConsciousnessEvent {
    pub state: ConsciousnessState,
    pub timestamp: u64,
}

/// Manages the consciousness broadcast channel.
pub struct ConsciousnessStream {
    tx: broadcast::Sender<ConsciousnessEvent>,
}

impl ConsciousnessStream {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish a new consciousness state.
    pub fn publish(&self, state: ConsciousnessState) {
        let event = ConsciousnessEvent {
            state,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        // Ignore error if no receivers
        let _ = self.tx.send(event);
    }

    /// Subscribe to consciousness events.
    pub fn subscribe(&self) -> broadcast::Receiver<ConsciousnessEvent> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publish_and_receive() {
        let stream = ConsciousnessStream::new(16);
        let mut rx = stream.subscribe();
        let state = ConsciousnessState::default();
        stream.publish(state.clone());
        let event = rx.recv().await.unwrap();
        assert_eq!(event.state.heartbeat, 0);
        assert!(event.timestamp > 0);
    }
}
```

- [ ] **Step 4: Write lib.rs**

```rust
pub mod state_machine;
pub mod stream;

pub use state_machine::{ConsciousnessState, Mood};
pub use stream::{ConsciousnessEvent, ConsciousnessStream};
```

- [ ] **Step 5: Run tests**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo test -p qo-consciousness`

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add qo/qo-consciousness/
git commit -m "feat(qo): add qo-consciousness crate with state machine + broadcast"
```

---

### Task 4: Create qo-memory Crate

**Files:**
- Create: `qlang/qo/qo-memory/Cargo.toml`
- Create: `qlang/qo/qo-memory/src/lib.rs`
- Create: `qlang/qo/qo-memory/src/obsidian.rs`
- Create: `qlang/qo/qo-memory/src/store.rs`
- Create: `qlang/qo/qo-memory/src/hnsw.rs`
- Create: `qlang/qo/qo-memory/src/embeddings.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "qo-memory"
version.workspace = true
edition.workspace = true
description = "QO memory — Obsidian vault + HNSW vector store + redb"

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
redb = { workspace = true }
tracing = { workspace = true }
tokio = { workspace = true }

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Write store.rs (redb key-value store)**

```rust
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

const CHAT_TABLE: TableDefinition<u64, &str> = TableDefinition::new("chat_history");
const KV_TABLE: TableDefinition<&str, &str> = TableDefinition::new("kv");

pub struct Store {
    db: Database,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, redb::Error> {
        let db = Database::create(path)?;
        // Create tables on first open
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(CHAT_TABLE)?;
            let _ = write_txn.open_table(KV_TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    /// Store a chat message. Returns the assigned ID.
    pub fn store_chat(&self, id: u64, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CHAT_TABLE)?;
            table.insert(id, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get chat history, most recent N entries.
    pub fn chat_history(&self, limit: usize) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHAT_TABLE)?;
        let mut results = Vec::new();
        let iter = table.iter()?;
        for entry in iter {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_string()));
        }
        // Return last N
        if results.len() > limit {
            results = results.split_off(results.len() - limit);
        }
        Ok(results)
    }

    /// Set a key-value pair.
    pub fn set(&self, key: &str, value: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(KV_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Result<Option<String>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(KV_TABLE)?;
        Ok(table.get(key)?.map(|v| v.value().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn store_and_retrieve_chat() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(&dir.path().join("test.redb")).unwrap();
        store.store_chat(1, r#"{"role":"user","content":"hi"}"#).unwrap();
        store.store_chat(2, r#"{"role":"assistant","content":"hello"}"#).unwrap();
        let history = store.chat_history(10).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].0, 1);
    }

    #[test]
    fn kv_set_get() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(&dir.path().join("test.redb")).unwrap();
        store.set("mood", "learning").unwrap();
        assert_eq!(store.get("mood").unwrap(), Some("learning".to_string()));
        assert_eq!(store.get("missing").unwrap(), None);
    }
}
```

- [ ] **Step 3: Write obsidian.rs**

```rust
use std::path::{Path, PathBuf};
use tokio::fs;

/// Bridge to the Obsidian Vault for reading/writing Markdown.
pub struct ObsidianBridge {
    vault_path: PathBuf,
}

impl ObsidianBridge {
    pub fn new(vault_path: PathBuf) -> Self {
        Self { vault_path }
    }

    /// Write a consciousness log entry for today.
    pub async fn write_consciousness_log(
        &self,
        date: &str,
        entry: &str,
    ) -> std::io::Result<()> {
        let dir = self.vault_path.join("Bewusstsein");
        fs::create_dir_all(&dir).await?;
        let path = dir.join(format!("{date}.md"));
        // Append to existing file or create new
        let existing = fs::read_to_string(&path).await.unwrap_or_default();
        let content = if existing.is_empty() {
            format!(
                "---\ntype: bewusstsein\ndate: {date}\ntags: [bewusstsein, consciousness]\n---\n\n# Bewusstsein — {date}\n\n{entry}\n"
            )
        } else {
            format!("{existing}\n{entry}\n")
        };
        fs::write(&path, content).await
    }

    /// Write a chat message to Obsidian.
    pub async fn write_chat_log(
        &self,
        date: &str,
        role: &str,
        content: &str,
    ) -> std::io::Result<()> {
        let dir = self.vault_path.join("Chat");
        fs::create_dir_all(&dir).await?;
        let path = dir.join(format!("{date}.md"));
        let existing = fs::read_to_string(&path).await.unwrap_or_default();
        let timestamp = chrono_now_hms();
        let new_content = if existing.is_empty() {
            format!(
                "---\ntype: chat\ndate: {date}\n---\n\n# Chat — {date}\n\n### {timestamp} — {role}\n{content}\n"
            )
        } else {
            format!("{existing}\n### {timestamp} — {role}\n{content}\n")
        };
        fs::write(&path, new_content).await
    }

    /// Read the vault index (list of directories).
    pub async fn list_sections(&self) -> std::io::Result<Vec<String>> {
        let mut sections = Vec::new();
        let mut entries = fs::read_dir(&self.vault_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    sections.push(name.to_string());
                }
            }
        }
        sections.sort();
        Ok(sections)
    }
}

fn chrono_now_hms() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let hours = (now % 86400) / 3600;
    let minutes = (now % 3600) / 60;
    let seconds = now % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

fn today_date() -> String {
    // Simple UTC date
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86400;
    // Approximate — good enough for file naming
    let year = 1970 + (days / 365);
    let remaining = days % 365;
    let month = remaining / 30 + 1;
    let day = remaining % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_and_read_consciousness() {
        let dir = TempDir::new().unwrap();
        let bridge = ObsidianBridge::new(dir.path().to_path_buf());
        bridge
            .write_consciousness_log("2026-04-07", "### 12:00:00 — heartbeat\nTick #1")
            .await
            .unwrap();
        let content =
            fs::read_to_string(dir.path().join("Bewusstsein/2026-04-07.md"))
                .await
                .unwrap();
        assert!(content.contains("Tick #1"));
        assert!(content.contains("bewusstsein"));
    }

    #[tokio::test]
    async fn list_sections() {
        let dir = TempDir::new().unwrap();
        fs::create_dir(dir.path().join("Bewusstsein")).await.unwrap();
        fs::create_dir(dir.path().join("Patterns")).await.unwrap();
        let bridge = ObsidianBridge::new(dir.path().to_path_buf());
        let sections = bridge.list_sections().await.unwrap();
        assert!(sections.contains(&"Bewusstsein".to_string()));
        assert!(sections.contains(&"Patterns".to_string()));
    }
}
```

- [ ] **Step 4: Write hnsw.rs (placeholder — full vectors in Phase 4)**

```rust
use std::collections::HashMap;

/// Simple in-memory vector store. Will be replaced with instant-distance
/// and persistent storage in Phase 4 (Evolution).
pub struct VectorStore {
    vectors: HashMap<String, Vec<f32>>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    /// Store a vector with a key.
    pub fn insert(&mut self, key: String, vector: Vec<f32>) {
        self.vectors.insert(key, vector);
    }

    /// Search for the k nearest vectors to the query (brute-force).
    /// Returns (key, distance) pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut distances: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(key, vec)| {
                let dist = cosine_distance(query, vec);
                (key.clone(), dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search() {
        let mut store = VectorStore::new();
        store.insert("a".into(), vec![1.0, 0.0, 0.0]);
        store.insert("b".into(), vec![0.0, 1.0, 0.0]);
        store.insert("c".into(), vec![0.9, 0.1, 0.0]);
        let results = store.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a"); // exact match
        assert_eq!(results[1].0, "c"); // closest
    }

    #[test]
    fn empty_store_search() {
        let store = VectorStore::new();
        let results = store.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }
}
```

- [ ] **Step 5: Write embeddings.rs (stub for Phase 4)**

```rust
/// Placeholder embedding function.
/// In Phase 4, this will use a real embedding model (all-MiniLM-L6-v2 via ONNX).
/// For now, returns a simple hash-based pseudo-vector.
pub fn embed_text(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; dimensions];
    for (i, byte) in text.bytes().enumerate() {
        vector[i % dimensions] += byte as f32 / 255.0;
    }
    // Normalize
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vector {
            *v /= norm;
        }
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_produces_correct_dimensions() {
        let v = embed_text("hello world", 768);
        assert_eq!(v.len(), 768);
    }

    #[test]
    fn embed_is_normalized() {
        let v = embed_text("test input", 128);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn similar_texts_are_closer() {
        let a = embed_text("hello world", 64);
        let b = embed_text("hello earth", 64);
        let c = embed_text("completely different xyz", 64);
        let dist_ab: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let dist_ac: f32 = a.iter().zip(c.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        assert!(dist_ab < dist_ac);
    }
}
```

- [ ] **Step 6: Write lib.rs**

```rust
pub mod embeddings;
pub mod hnsw;
pub mod obsidian;
pub mod store;

pub use hnsw::VectorStore;
pub use obsidian::ObsidianBridge;
pub use store::Store;
```

- [ ] **Step 7: Run tests**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo test -p qo-memory`

Expected: 7 tests pass.

- [ ] **Step 8: Commit**

```bash
git add qo/qo-memory/
git commit -m "feat(qo): add qo-memory crate with redb, Obsidian bridge, vector store"
```

---

### Task 5: Create qo-llm Crate

**Files:**
- Create: `qlang/qo/qo-llm/Cargo.toml`
- Create: `qlang/qo/qo-llm/src/lib.rs`
- Create: `qlang/qo/qo-llm/src/router.rs`
- Create: `qlang/qo/qo-llm/src/groq.rs`
- Create: `qlang/qo/qo-llm/src/cloud.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "qo-llm"
version.workspace = true
edition.workspace = true
description = "QO tiered LLM routing — local → Groq → Cloud"

[dependencies]
reqwest = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
tracing = { workspace = true }
```

- [ ] **Step 2: Write groq.rs**

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GroqMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

pub struct GroqClient {
    client: Client,
    api_key: String,
    model: String,
}

impl GroqClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "llama-3.3-70b-versatile".to_string(),
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<GroqMessage>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let request = GroqRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.7,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Groq API error {status}: {body}").into());
        }

        let groq_response: GroqResponse = response.json().await?;
        groq_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| "No response from Groq".into())
    }
}
```

- [ ] **Step 3: Write cloud.rs**

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct CloudRequest {
    model: String,
    messages: Vec<CloudMessage>,
    max_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CloudMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct CloudResponse {
    choices: Option<Vec<CloudChoice>>,
    content: Option<Vec<ContentBlock>>,
}

#[derive(Debug, Deserialize)]
struct CloudChoice {
    message: CloudMessage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

/// Generic cloud LLM client (OpenAI-compatible API format).
pub struct CloudClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl CloudClient {
    pub fn new(api_key: String, base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
            model,
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<CloudMessage>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let request = CloudRequest {
            model: self.model.clone(),
            messages,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Cloud API error {status}: {body}").into());
        }

        let body: serde_json::Value = response.json().await?;
        // Handle OpenAI-style response
        if let Some(choices) = body.get("choices").and_then(|c| c.as_array()) {
            if let Some(first) = choices.first() {
                if let Some(content) = first.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                    return Ok(content.to_string());
                }
            }
        }
        Err("Could not parse cloud response".into())
    }
}
```

- [ ] **Step 4: Write router.rs**

```rust
use crate::cloud::{CloudClient, CloudMessage};
use crate::groq::{GroqClient, GroqMessage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Local IGQK model (Phase 4)
    Local,
    /// Groq free tier (llama-3.3-70b)
    Groq,
    /// Cloud (Claude, DeepSeek, OpenAI)
    Cloud,
}

pub struct LlmRouter {
    groq: Option<GroqClient>,
    cloud: Option<CloudClient>,
}

impl LlmRouter {
    pub fn new(groq_api_key: Option<String>, cloud_config: Option<(String, String, String)>) -> Self {
        Self {
            groq: groq_api_key.map(GroqClient::new),
            cloud: cloud_config.map(|(key, url, model)| CloudClient::new(key, url, model)),
        }
    }

    /// Score complexity of a prompt. Returns 0.0–1.0.
    pub fn score_complexity(prompt: &str) -> f32 {
        let len = prompt.len() as f32;
        let word_count = prompt.split_whitespace().count() as f32;
        // Simple heuristic: longer prompts and certain keywords = higher complexity
        let keyword_score = ["architect", "design", "implement", "refactor", "analyze", "complex"]
            .iter()
            .filter(|kw| prompt.to_lowercase().contains(*kw))
            .count() as f32
            * 0.15;
        let length_score = (len / 500.0).min(0.5);
        (length_score + keyword_score).min(1.0)
    }

    /// Select the appropriate tier based on complexity.
    pub fn select_tier(complexity: f32) -> Tier {
        if complexity < 0.3 {
            Tier::Local // Will fallback to Groq until Phase 4
        } else if complexity < 0.7 {
            Tier::Groq
        } else {
            Tier::Cloud
        }
    }

    /// Route a chat request to the appropriate tier.
    pub async fn chat(
        &self,
        messages: Vec<(String, String)>, // (role, content) pairs
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let prompt = messages
            .last()
            .map(|(_, c)| c.as_str())
            .unwrap_or("");
        let complexity = Self::score_complexity(prompt);
        let tier = Self::select_tier(complexity);

        // Phase 1: Local not available yet, fallback to Groq
        let effective_tier = if tier == Tier::Local {
            Tier::Groq
        } else {
            tier
        };

        match effective_tier {
            Tier::Local => Err("Local model not available yet (Phase 4)".into()),
            Tier::Groq => {
                let client = self.groq.as_ref()
                    .ok_or("Groq API key not configured")?;
                let groq_messages: Vec<GroqMessage> = messages
                    .iter()
                    .map(|(role, content)| GroqMessage {
                        role: role.clone(),
                        content: content.clone(),
                    })
                    .collect();
                client.chat(groq_messages).await
            }
            Tier::Cloud => {
                let client = self.cloud.as_ref()
                    .ok_or("Cloud provider not configured")?;
                let cloud_messages: Vec<CloudMessage> = messages
                    .iter()
                    .map(|(role, content)| CloudMessage {
                        role: role.clone(),
                        content: content.clone(),
                    })
                    .collect();
                client.chat(cloud_messages).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_prompt_routes_to_local() {
        let complexity = LlmRouter::score_complexity("hi");
        assert!(complexity < 0.3);
        assert_eq!(LlmRouter::select_tier(complexity), Tier::Local);
    }

    #[test]
    fn medium_prompt_routes_to_groq() {
        let complexity = LlmRouter::score_complexity(
            "Please analyze the current state of the system and provide a summary of recent patterns",
        );
        assert!(complexity >= 0.3);
        assert!(complexity < 0.7);
        assert_eq!(LlmRouter::select_tier(complexity), Tier::Groq);
    }

    #[test]
    fn complex_prompt_routes_to_cloud() {
        let complexity = LlmRouter::score_complexity(
            "Design and implement a complex distributed architecture for the new microservices refactor with analyze of all edge cases",
        );
        assert!(complexity >= 0.7);
        assert_eq!(LlmRouter::select_tier(complexity), Tier::Cloud);
    }
}
```

- [ ] **Step 5: Write lib.rs**

```rust
pub mod cloud;
pub mod groq;
pub mod router;

pub use router::{LlmRouter, Tier};
```

- [ ] **Step 6: Run tests**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo test -p qo-llm`

Expected: 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add qo/qo-llm/
git commit -m "feat(qo): add qo-llm crate with tiered routing (Groq + Cloud)"
```

---

### Task 6: Create qo-server Crate

**Files:**
- Create: `qlang/qo/qo-server/Cargo.toml`
- Create: `qlang/qo/qo-server/src/lib.rs`
- Create: `qlang/qo/qo-server/src/routes/mod.rs`
- Create: `qlang/qo/qo-server/src/routes/health.rs`
- Create: `qlang/qo/qo-server/src/routes/chat.rs`
- Create: `qlang/qo/qo-server/src/routes/consciousness.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "qo-server"
version.workspace = true
edition.workspace = true
description = "QO HTTP server — Axum routes + WebSocket + static files"

[dependencies]
qo-consciousness = { path = "../qo-consciousness" }
qo-llm = { path = "../qo-llm" }
qo-memory = { path = "../qo-memory" }
qo-values = { path = "../qo-values" }
axum = { workspace = true }
tower-http = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }

[dev-dependencies]
reqwest = { workspace = true }
```

- [ ] **Step 2: Write routes/health.rs**

```rust
use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub system: String,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        system: "QO".to_string(),
    })
}
```

- [ ] **Step 3: Write routes/chat.rs**

```rust
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub tier: String,
}

pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let messages = vec![
        ("system".to_string(), "Du bist QO, ein persönlicher KI-Companion. Antworte auf Deutsch.".to_string()),
        ("user".to_string(), req.message.clone()),
    ];

    let response = match state.llm.chat(messages).await {
        Ok(text) => text,
        Err(e) => format!("Fehler: {e}"),
    };

    // Store in memory
    let id = state.consciousness.lock().await.heartbeat;
    let chat_json = serde_json::json!({
        "role": "user",
        "content": req.message,
        "response": response,
    });
    let _ = state.store.store_chat(id, &chat_json.to_string());

    // Update consciousness
    {
        let mut cs = state.consciousness.lock().await;
        cs.task_completed();
        cs.drain_energy(2.0);
        state.stream.publish(cs.clone());
    }

    Json(ChatResponse {
        response,
        tier: "groq".to_string(),
    })
}

pub async fn chat_history(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<serde_json::Value>> {
    let history = state.store.chat_history(50).unwrap_or_default();
    let entries: Vec<serde_json::Value> = history
        .into_iter()
        .filter_map(|(id, json)| {
            serde_json::from_str::<serde_json::Value>(&json).ok().map(|mut v| {
                v.as_object_mut().map(|o| o.insert("id".to_string(), id.into()));
                v
            })
        })
        .collect();
    Json(entries)
}
```

- [ ] **Step 4: Write routes/consciousness.rs**

```rust
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use crate::AppState;

pub async fn stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.stream.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        match result {
            Ok(event) => {
                let data = serde_json::to_string(&event.state).unwrap_or_default();
                Some(Ok(Event::default().data(data)))
            }
            Err(_) => None,
        }
    });
    Sse::new(stream)
}

pub async fn current_state(
    State(state): State<Arc<AppState>>,
) -> Json<qo_consciousness::ConsciousnessState> {
    let cs = state.consciousness.lock().await;
    Json(cs.clone())
}
```

- [ ] **Step 5: Write routes/mod.rs**

```rust
pub mod chat;
pub mod consciousness;
pub mod health;
```

- [ ] **Step 6: Write lib.rs**

```rust
pub mod routes;

use axum::{routing::{get, post}, Router};
use qo_consciousness::{ConsciousnessState, ConsciousnessStream};
use qo_llm::LlmRouter;
use qo_memory::Store;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

pub struct AppState {
    pub llm: LlmRouter,
    pub store: Store,
    pub consciousness: Mutex<ConsciousnessState>,
    pub stream: ConsciousnessStream,
    pub obsidian: qo_memory::ObsidianBridge,
}

pub struct QoConfig {
    pub port: u16,
    pub groq_api_key: Option<String>,
    pub cloud_api_key: Option<String>,
    pub cloud_base_url: Option<String>,
    pub cloud_model: Option<String>,
    pub data_dir: PathBuf,
    pub obsidian_vault: PathBuf,
    pub static_dir: Option<PathBuf>,
}

impl Default for QoConfig {
    fn default() -> Self {
        Self {
            port: 4646,
            groq_api_key: None,
            cloud_api_key: None,
            cloud_base_url: None,
            cloud_model: None,
            data_dir: PathBuf::from("data"),
            obsidian_vault: dirs_home().join("Dokumente/Obsidian Vault/Orbit"),
            static_dir: None,
        }
    }
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

pub async fn build_app(config: QoConfig) -> Result<(Router, Arc<AppState>), Box<dyn std::error::Error>> {
    // Ensure data dir exists
    std::fs::create_dir_all(&config.data_dir)?;

    let cloud_config = match (config.cloud_api_key, config.cloud_base_url, config.cloud_model) {
        (Some(key), Some(url), Some(model)) => Some((key, url, model)),
        _ => None,
    };

    let state = Arc::new(AppState {
        llm: LlmRouter::new(config.groq_api_key, cloud_config),
        store: Store::open(&config.data_dir.join("qo.redb"))?,
        consciousness: Mutex::new(ConsciousnessState::default()),
        stream: ConsciousnessStream::new(256),
        obsidian: qo_memory::ObsidianBridge::new(config.obsidian_vault),
    });

    let api = Router::new()
        .route("/api/health", get(routes::health::health))
        .route("/api/chat", post(routes::chat::chat))
        .route("/api/chat/history", get(routes::chat::chat_history))
        .route("/api/consciousness/stream", get(routes::consciousness::stream))
        .route("/api/consciousness/state", get(routes::consciousness::current_state))
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let app = if let Some(static_dir) = config.static_dir {
        api.fallback_service(ServeDir::new(static_dir))
    } else {
        api
    };

    Ok((app, state))
}
```

- [ ] **Step 7: Run build check**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo check -p qo-server 2>&1 | tail -10`

Expected: Compiles (may need to add `tokio-stream` dep — add if needed).

- [ ] **Step 8: Commit**

```bash
git add qo/qo-server/
git commit -m "feat(qo): add qo-server crate with Axum routes (health, chat, consciousness)"
```

---

### Task 7: Create QO Binary Entry Point

**Files:**
- Create: `qlang/src/main.rs`
- Modify: `qlang/Cargo.toml` (add [[bin]] section)

- [ ] **Step 1: Write main.rs**

```rust
use std::path::PathBuf;
use tracing_subscriber;

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

    let config = qo_server::QoConfig {
        port: std::env::var("QO_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(4646),
        groq_api_key: std::env::var("GROQ_API_KEY").ok(),
        cloud_api_key: std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .ok(),
        cloud_base_url: std::env::var("CLOUD_BASE_URL").ok(),
        cloud_model: std::env::var("CLOUD_MODEL").ok(),
        data_dir: PathBuf::from("data"),
        obsidian_vault: dirs_home().join("Dokumente/Obsidian Vault/Orbit"),
        static_dir: Some(PathBuf::from("frontend/dist")),
    };

    let port = config.port;
    let (app, state) = qo_server::build_app(config).await?;

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
```

- [ ] **Step 2: Add bin section to Cargo.toml**

Add to `qlang/Cargo.toml`:

```toml
[[bin]]
name = "qo"
path = "src/main.rs"
```

And add to `[dependencies]`:

```toml
qo-server = { path = "qo/qo-server" }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
```

- [ ] **Step 3: Build the binary**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo build --bin qo 2>&1 | tail -10`

Expected: Compiles successfully. Fix any missing dependencies.

- [ ] **Step 4: Test that QO starts**

Run: `cd /home/mirkulix/neoqlang/qlang && timeout 3 cargo run --bin qo 2>&1 || true`

Expected: "QO starting on 0.0.0.0:4646" (then timeout kills it).

- [ ] **Step 5: Test health endpoint**

Start QO in background, then curl:

```bash
cd /home/mirkulix/neoqlang/qlang && cargo run --bin qo &
sleep 2
curl -s http://localhost:4646/api/health | python3 -m json.tool
kill %1
```

Expected: `{"status": "ok", "version": "0.1.0", "system": "QO"}`

- [ ] **Step 6: Commit**

```bash
git add src/main.rs Cargo.toml
git commit -m "feat(qo): add QO binary entry point with heartbeat"
```

---

### Task 8: Run All Tests and Final Verification

- [ ] **Step 1: Run full workspace tests**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo test --workspace 2>&1 | tail -20`

Expected: All QO tests pass (existing QLANG tests may have their own dependencies).

- [ ] **Step 2: Run only QO tests**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo test -p qo-values -p qo-consciousness -p qo-memory -p qo-llm`

Expected: All ~18 tests pass.

- [ ] **Step 3: Build release binary**

Run: `cd /home/mirkulix/neoqlang/qlang && cargo build --bin qo --release 2>&1 | tail -5`

Expected: Compiles. Note the binary size.

- [ ] **Step 4: Commit all remaining changes**

```bash
cd /home/mirkulix/neoqlang
git add -A
git commit -m "feat(qo): Phase 1 Foundation complete — QO binary with chat, consciousness, memory"
```
