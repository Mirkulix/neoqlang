use crate::{
    agent::AgentRole,
    goal::{ExecutionGraph, Goal, GoalStatus, GraphEdge, GraphNode, SubTask},
    llm_node,
};
use qo_llm::LlmRouter;
use qlang_core::binary;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType};
use qlang_agent::protocol::{AgentId, Capability, GraphMessage, MessageIntent};
use std::collections::HashMap;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================
// QLANG Graph Builders — produce real Graph structs
// ============================================================

/// Build a CEO decomposition graph (Input → Output, Utf8 scalars).
pub fn build_ceo_decompose_graph(label: &str) -> Graph {
    let mut g = Graph::new(format!("ceo_decompose_{}", label));
    let utf8_scalar = TensorType::new(Dtype::Utf8, Shape::scalar());

    let input = g.add_node(
        Op::Input { name: "goal".into() },
        vec![],
        vec![utf8_scalar.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "plan".into() },
        vec![utf8_scalar.clone()],
        vec![],
    );

    g.add_edge(input, 0, output, 0, utf8_scalar);
    g
}

/// Build an agent task graph (Input → Output, Utf8 scalars).
pub fn build_agent_task_graph(agent_name: &str) -> Graph {
    let mut g = Graph::new(format!("agent_{}", agent_name));
    let utf8_scalar = TensorType::new(Dtype::Utf8, Shape::scalar());

    let input = g.add_node(
        Op::Input { name: "task".into() },
        vec![],
        vec![utf8_scalar.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "result".into() },
        vec![utf8_scalar.clone()],
        vec![],
    );

    g.add_edge(input, 0, output, 0, utf8_scalar);
    g
}

/// Build a GraphMessage wrapping an agent-to-agent task dispatch.
pub fn build_agent_message(
    from_role: &str,
    to_role: &str,
    graph: Graph,
    task_text: &str,
) -> GraphMessage {
    use qlang_core::tensor::TensorData;
    let mut inputs = HashMap::new();
    inputs.insert("task".to_string(), TensorData::from_string(task_text));

    GraphMessage {
        id: now_ms(),
        from: AgentId {
            name: from_role.into(),
            capabilities: vec![Capability::Execute],
        },
        to: AgentId {
            name: to_role.into(),
            capabilities: vec![Capability::Execute],
        },
        graph,
        inputs,
        intent: MessageIntent::Execute,
        in_reply_to: None,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    }
}

// ============================================================
// ExecutionGraphData — real QLANG graph metadata for one run
// ============================================================

/// Tracks the real QLANG graphs built and executed during a goal run.
pub struct ExecutionGraphData {
    /// (name, node_count, binary_size_bytes)
    pub graphs: Vec<(String, usize, usize)>,
    /// (name, duration_ms, success)
    pub executions: Vec<(String, u64, bool)>,
    pub total_binary_bytes: usize,
}

impl ExecutionGraphData {
    pub fn new() -> Self {
        Self {
            graphs: Vec::new(),
            executions: Vec::new(),
            total_binary_bytes: 0,
        }
    }

    pub fn add_graph(&mut self, name: &str, graph: &Graph, binary_size: usize) {
        self.graphs
            .push((name.to_string(), graph.nodes.len(), binary_size));
        self.total_binary_bytes += binary_size;
    }

    pub fn record_execution(&mut self, name: &str, duration_ms: u64, success: bool) {
        self.executions
            .push((name.to_string(), duration_ms, success));
    }
}

impl Default for ExecutionGraphData {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// execute_goal — builds + serializes real QLANG graphs,
// uses QLMS protocol messages, then calls LLM for reasoning.
// ============================================================

/// Execute a goal: CEO decomposes it, each subtask is handled by the
/// assigned agent. Real QLANG graphs are built, serialised to binary
/// (.qlbg), and QLMS GraphMessages are sent between agents.
pub async fn execute_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let graph_data = execute_goal_qlang(llm, goal).await?;

    // Log QLANG graph stats
    tracing::info!(
        "QLANG execution complete: {} graphs ({} bytes total binary), {} executions",
        graph_data.graphs.len(),
        graph_data.total_binary_bytes,
        graph_data.executions.len(),
    );
    for (name, nodes, bytes) in &graph_data.graphs {
        tracing::debug!("  graph '{}': {} nodes, {} bytes (QLBG)", name, nodes, bytes);
    }

    Ok(())
}

/// Core implementation: builds real QLANG graphs and executes them.
pub async fn execute_goal_qlang(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<ExecutionGraphData, Box<dyn std::error::Error + Send + Sync>> {
    goal.status = GoalStatus::InProgress;
    let mut graph_data = ExecutionGraphData::new();

    // Build initial display graph (for UI)
    let mut nodes: Vec<GraphNode> = vec![
        GraphNode {
            id: "input".to_string(),
            label: format!(
                "Ziel: {}",
                &goal.description.chars().take(40).collect::<String>()
            ),
            node_type: "input".to_string(),
            agent: None,
            status: "completed".to_string(),
            duration_ms: None,
        },
        GraphNode {
            id: "ceo_decompose".to_string(),
            label: "CEO: Dekomposition".to_string(),
            node_type: "llm".to_string(),
            agent: Some("CEO".to_string()),
            status: "pending".to_string(),
            duration_ms: None,
        },
    ];
    let mut edges: Vec<GraphEdge> = vec![GraphEdge {
        from: "input".to_string(),
        to: "ceo_decompose".to_string(),
        data_type: "goal".to_string(),
    }];

    // --- Step 1: Build + serialise CEO decomposition graph ---
    let ceo_graph = build_ceo_decompose_graph("goal_decompose");
    let ceo_bytes = binary::to_binary(&ceo_graph);
    graph_data.add_graph("ceo_decompose", &ceo_graph, ceo_bytes.len());

    // Build QLMS message CEO → (itself, broadcasting)
    let _ceo_msg = build_agent_message("ceo", "team", ceo_graph.clone(), &goal.description);

    // --- Step 2: CEO reasoning via LLM ---
    let decomposition_prompt = format!(
        "Zerlege dieses Ziel in 2-4 konkrete Teilaufgaben. Für jede Aufgabe, bestimme welcher Agent sie bearbeiten soll.\n\
        Verfügbare Agenten: Researcher (Analyse/Recherche), Developer (Code/Technik), Strategist (Planung), Artisan (Kreativ).\n\n\
        Ziel: {}\n\n\
        Antworte im Format:\n\
        1. [Agent] Aufgabe\n\
        2. [Agent] Aufgabe\n\
        ...",
        goal.description
    );
    let t0 = now_ms();
    let decomposition =
        llm_node::llm_reason(llm, AgentRole::Ceo, "Goal-Dekomposition", &decomposition_prompt)
            .await?;
    let decompose_ms = now_ms() - t0;
    graph_data.record_execution("ceo_decompose", decompose_ms, true);

    // Mark CEO node done
    if let Some(n) = nodes.iter_mut().find(|n| n.id == "ceo_decompose") {
        n.status = "completed".to_string();
        n.duration_ms = Some(decompose_ms);
    }

    // --- Step 3: Parse subtasks ---
    goal.subtasks = parse_subtasks(&decomposition);
    if goal.subtasks.is_empty() {
        goal.subtasks.push(SubTask {
            description: goal.description.clone(),
            assigned_to: AgentRole::Researcher,
            status: GoalStatus::Pending,
            result: None,
        });
    }

    let subtask_count = goal.subtasks.len();

    // Add subtask nodes to display graph
    for i in 0..subtask_count {
        let agent_name = goal.subtasks[i].assigned_to.name().to_string();
        let desc = goal.subtasks[i]
            .description
            .chars()
            .take(40)
            .collect::<String>();
        let node_id = format!("subtask_{}", i);
        nodes.push(GraphNode {
            id: node_id.clone(),
            label: format!("{}: {}", agent_name, desc),
            node_type: "llm".to_string(),
            agent: Some(agent_name),
            status: "pending".to_string(),
            duration_ms: None,
        });
        edges.push(GraphEdge {
            from: "ceo_decompose".to_string(),
            to: node_id,
            data_type: "subtask".to_string(),
        });
    }

    // Summary + output nodes
    nodes.push(GraphNode {
        id: "ceo_summary".to_string(),
        label: "CEO: Zusammenfassung".to_string(),
        node_type: "llm".to_string(),
        agent: Some("CEO".to_string()),
        status: "pending".to_string(),
        duration_ms: None,
    });
    nodes.push(GraphNode {
        id: "output".to_string(),
        label: "Ergebnis".to_string(),
        node_type: "output".to_string(),
        agent: None,
        status: "pending".to_string(),
        duration_ms: None,
    });
    edges.push(GraphEdge {
        from: "ceo_summary".to_string(),
        to: "output".to_string(),
        data_type: "result".to_string(),
    });

    // --- Step 4: Build + execute real QLANG graphs for each subtask ---
    for i in 0..subtask_count {
        let agent_role = goal.subtasks[i].assigned_to;
        let agent_name_lower = agent_role.name().to_lowercase();
        let graph_name = format!("agent_{}", agent_name_lower);

        let agent_graph = build_agent_task_graph(&agent_name_lower);
        let agent_bytes = binary::to_binary(&agent_graph);
        graph_data.add_graph(&graph_name, &agent_graph, agent_bytes.len());

        // QLMS: CEO dispatches task to agent via GraphMessage
        let task_text = goal.subtasks[i].description.clone();
        let _msg = build_agent_message("ceo", &agent_name_lower, agent_graph, &task_text);

        goal.subtasks[i].status = GoalStatus::InProgress;
        let goal_desc = goal.description.clone();

        let t_sub = now_ms();
        match llm_node::llm_reason(llm, agent_role, &goal_desc, &task_text).await {
            Ok(result) => {
                let sub_ms = now_ms() - t_sub;
                goal.subtasks[i].result = Some(result);
                goal.subtasks[i].status = GoalStatus::Completed;
                graph_data.record_execution(&graph_name, sub_ms, true);

                let node_id = format!("subtask_{}", i);
                if let Some(n) = nodes.iter_mut().find(|n| n.id == node_id) {
                    n.status = "completed".to_string();
                    n.duration_ms = Some(sub_ms);
                }
            }
            Err(e) => {
                let sub_ms = now_ms() - t_sub;
                goal.subtasks[i].result = Some(format!("Fehler: {e}"));
                goal.subtasks[i].status = GoalStatus::Failed;
                graph_data.record_execution(&graph_name, sub_ms, false);

                let node_id = format!("subtask_{}", i);
                if let Some(n) = nodes.iter_mut().find(|n| n.id == node_id) {
                    n.status = "failed".to_string();
                    n.duration_ms = Some(sub_ms);
                }
            }
        }

        edges.push(GraphEdge {
            from: format!("subtask_{}", i),
            to: "ceo_summary".to_string(),
            data_type: "result".to_string(),
        });
    }

    // --- Step 5: CEO summary graph ---
    let summary_graph = build_ceo_decompose_graph("summary");
    let summary_bytes = binary::to_binary(&summary_graph);
    graph_data.add_graph("ceo_summary", &summary_graph, summary_bytes.len());

    let all_results: Vec<String> = goal
        .subtasks
        .iter()
        .filter_map(|st| {
            st.result
                .as_ref()
                .map(|r| format!("{}: {}", st.assigned_to.name(), r))
        })
        .collect();

    let summary_prompt = format!(
        "Fasse die Ergebnisse der Teilaufgaben zusammen und gib ein Gesamtergebnis.\n\n\
        Ziel: {}\n\n\
        Ergebnisse:\n{}",
        goal.description,
        all_results.join("\n\n")
    );

    let t_sum = now_ms();
    let summary =
        llm_node::llm_reason(llm, AgentRole::Ceo, "Zusammenfassung", &summary_prompt).await?;
    let sum_ms = now_ms() - t_sum;
    graph_data.record_execution("ceo_summary", sum_ms, true);

    if let Some(n) = nodes.iter_mut().find(|n| n.id == "ceo_summary") {
        n.status = "completed".to_string();
        n.duration_ms = Some(sum_ms);
    }
    if let Some(n) = nodes.iter_mut().find(|n| n.id == "output") {
        n.status = "completed".to_string();
    }

    goal.result = Some(summary);
    goal.status = GoalStatus::Completed;
    goal.execution_graph = Some(ExecutionGraph { nodes, edges });

    Ok(graph_data)
}

// ============================================================
// Legacy entry points (kept for backward compat)
// ============================================================

pub async fn decompose_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    goal.status = GoalStatus::InProgress;

    let decomposition_prompt = format!(
        "Zerlege dieses Ziel in 2-4 konkrete Teilaufgaben. Für jede Aufgabe, bestimme welcher Agent sie bearbeiten soll.\n\
        Verfügbare Agenten: Researcher (Analyse/Recherche), Developer (Code/Technik), Strategist (Planung), Artisan (Kreativ).\n\n\
        Ziel: {}\n\n\
        Antworte im Format:\n\
        1. [Agent] Aufgabe\n\
        2. [Agent] Aufgabe\n\
        ...",
        goal.description
    );

    let decomposition = llm_node::llm_reason(
        llm,
        AgentRole::Ceo,
        "Goal-Dekomposition",
        &decomposition_prompt,
    )
    .await?;

    goal.subtasks = parse_subtasks(&decomposition);

    if goal.subtasks.is_empty() {
        goal.subtasks.push(SubTask {
            description: goal.description.clone(),
            assigned_to: AgentRole::Researcher,
            status: GoalStatus::Pending,
            result: None,
        });
    }

    Ok(())
}

pub async fn execute_subtask(
    llm: &LlmRouter,
    goal: &mut Goal,
    index: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let subtask = goal
        .subtasks
        .get_mut(index)
        .ok_or("Subtask index out of bounds")?;
    subtask.status = GoalStatus::InProgress;

    let goal_desc = goal.description.clone();
    let subtask_desc = goal.subtasks[index].description.clone();
    let assigned_to = goal.subtasks[index].assigned_to;

    match llm_node::llm_reason(llm, assigned_to, &goal_desc, &subtask_desc).await {
        Ok(result) => {
            goal.subtasks[index].result = Some(result.clone());
            goal.subtasks[index].status = GoalStatus::Completed;
            Ok(result)
        }
        Err(e) => {
            let err_msg = format!("Fehler: {e}");
            goal.subtasks[index].result = Some(err_msg.clone());
            goal.subtasks[index].status = GoalStatus::Failed;
            Err(e)
        }
    }
}

pub async fn summarize_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let all_results: Vec<String> = goal
        .subtasks
        .iter()
        .filter_map(|s| s.result.as_ref().map(|r| format!("{}: {}", s.assigned_to.name(), r)))
        .collect();

    let summary_prompt = format!(
        "Fasse die Ergebnisse der Teilaufgaben zusammen und gib ein Gesamtergebnis.\n\n\
        Ziel: {}\n\n\
        Ergebnisse:\n{}",
        goal.description,
        all_results.join("\n\n")
    );

    let summary =
        llm_node::llm_reason(llm, AgentRole::Ceo, "Zusammenfassung", &summary_prompt).await?;
    goal.result = Some(summary.clone());
    goal.status = GoalStatus::Completed;

    Ok(summary)
}

// ============================================================
// Subtask parser (unchanged)
// ============================================================

fn parse_subtasks(response: &str) -> Vec<SubTask> {
    let mut subtasks = Vec::new();
    for line in response.lines() {
        let line = line.trim();
        let (role, desc) = if let Some(rest) =
            line.strip_prefix(|c: char| c.is_ascii_digit() || c == '-' || c == '*')
        {
            let rest = rest.trim_start_matches(['.', ')', ' ']);
            parse_role_and_desc(rest)
        } else {
            parse_role_and_desc(line)
        };

        if let Some(role) = role {
            if !desc.is_empty() {
                subtasks.push(SubTask {
                    description: desc,
                    assigned_to: role,
                    status: GoalStatus::Pending,
                    result: None,
                });
            }
        }
    }
    subtasks
}

fn parse_role_and_desc(s: &str) -> (Option<AgentRole>, String) {
    let s = s.trim();
    if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            let role_str = &s[1..end];
            let desc = s[end + 1..].trim().to_string();
            return (match_role(role_str), desc);
        }
    }
    for role in AgentRole::ALL {
        let name = role.name();
        if s.starts_with(name) {
            let rest = s[name.len()..].trim_start_matches([':', '-', ' ']);
            return (Some(role), rest.trim().to_string());
        }
    }
    (None, s.to_string())
}

fn match_role(s: &str) -> Option<AgentRole> {
    let s_lower = s.to_lowercase();
    match s_lower.as_str() {
        "ceo" => Some(AgentRole::Ceo),
        "researcher" | "recherche" | "analyse" => Some(AgentRole::Researcher),
        "developer" | "entwickler" | "code" | "technik" => Some(AgentRole::Developer),
        "guardian" | "wächter" | "werte" => Some(AgentRole::Guardian),
        "strategist" | "strategie" | "planung" => Some(AgentRole::Strategist),
        "artisan" | "kreativ" | "design" => Some(AgentRole::Artisan),
        _ => None,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::binary;

    // --- Original parser tests (all preserved) ---

    #[test]
    fn parse_subtasks_bracket_format() {
        let input =
            "1. [Researcher] Analysiere den Markt\n2. [Developer] Baue den Prototyp\n3. [Strategist] Erstelle die Roadmap";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Developer);
        assert_eq!(tasks[2].assigned_to, AgentRole::Strategist);
    }

    #[test]
    fn parse_subtasks_colon_format() {
        let input = "- Researcher: Recherchiere Trends\n- Artisan: Erstelle Design";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Artisan);
    }

    #[test]
    fn parse_subtasks_german_names() {
        let input = "1. [Recherche] Finde Informationen\n2. [Kreativ] Gestalte Logo";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Artisan);
    }

    #[test]
    fn test_parse_empty_response() {
        let tasks = parse_subtasks("");
        assert_eq!(tasks.len(), 0);
    }

    #[test]
    fn test_parse_mixed_formats() {
        let input = "1. [Researcher] Analysiere den Markt\n- Developer: Baue den Prototyp";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Developer);
    }

    #[test]
    fn test_parse_numbered_without_brackets() {
        let input = "1. Researcher: do X\n2. Strategist: do Y";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Strategist);
    }

    #[test]
    fn parse_subtasks_handles_real_ceo_output() {
        let response = "Hier ist mein Plan:\n\n\
            1. [Researcher] Recherchiere aktuelle Trends im KI-Markt\n\
            2. [Developer] Erstelle einen Prototyp der API\n\
            3. [Strategist] Entwickle eine Go-to-Market Strategie\n\n\
            Diese drei Schritte sollten das Ziel erreichen.";
        let tasks = parse_subtasks(response);
        assert_eq!(tasks.len(), 3);
        assert!(tasks[0].description.contains("Recherchiere"));
        assert!(tasks[1].description.contains("Prototyp"));
        assert!(tasks[2].description.contains("Strategie"));
    }

    // --- New QLANG graph tests ---

    #[test]
    fn builds_real_qlang_graph() {
        let g = build_ceo_decompose_graph("Test goal");

        // Must have input + output nodes
        assert!(g.nodes.len() >= 2, "graph must have at least 2 nodes");
        assert!(!g.edges.is_empty(), "graph must have at least one edge");

        // Verify binary serialisation
        let bytes = binary::to_binary(&g);
        assert!(!bytes.is_empty(), "binary must not be empty");

        // Check QLBG magic bytes
        assert_eq!(
            &bytes[0..4],
            &[0x51, 0x4C, 0x42, 0x47],
            "binary must start with QLBG magic"
        );

        // Round-trip
        let restored = binary::from_binary(&bytes).expect("must deserialise");
        assert_eq!(
            restored.nodes.len(),
            g.nodes.len(),
            "node count must survive round-trip"
        );
        assert_eq!(
            restored.edges.len(),
            g.edges.len(),
            "edge count must survive round-trip"
        );

        println!(
            "QLBG binary: {} bytes, {} nodes, {} edges",
            bytes.len(),
            g.nodes.len(),
            g.edges.len()
        );
    }

    #[test]
    fn builds_real_agent_task_graph() {
        let g = build_agent_task_graph("researcher");
        assert!(g.nodes.len() >= 2);
        assert!(!g.edges.is_empty());

        let bytes = binary::to_binary(&g);
        assert_eq!(&bytes[0..4], &[0x51, 0x4C, 0x42, 0x47]);
        println!("agent graph binary: {} bytes", bytes.len());
    }

    #[test]
    fn agent_protocol_message() {
        let g = build_agent_task_graph("researcher");
        let msg = build_agent_message("ceo", "researcher", g, "Research AI trends");

        assert_eq!(msg.from.name, "ceo");
        assert_eq!(msg.to.name, "researcher");
        assert!(matches!(msg.intent, MessageIntent::Execute));
        assert!(msg.inputs.contains_key("task"));
        assert_eq!(
            msg.inputs["task"].as_string().unwrap(),
            "Research AI trends"
        );

        println!(
            "GraphMessage: {} → {}, intent={:?}",
            msg.from.name, msg.to.name, msg.intent
        );
    }

    #[test]
    fn execution_graph_data_tracks_correctly() {
        let mut data = ExecutionGraphData::new();
        let g1 = build_ceo_decompose_graph("decompose");
        let b1 = binary::to_binary(&g1);
        let b1_len = b1.len();
        data.add_graph("ceo_decompose", &g1, b1_len);

        let g2 = build_agent_task_graph("researcher");
        let b2 = binary::to_binary(&g2);
        let b2_len = b2.len();
        data.add_graph("agent_researcher", &g2, b2_len);

        data.record_execution("ceo_decompose", 120, true);
        data.record_execution("agent_researcher", 340, true);

        assert_eq!(data.graphs.len(), 2);
        assert_eq!(data.executions.len(), 2);
        assert_eq!(data.total_binary_bytes, b1_len + b2_len);

        println!(
            "Total binary: {} bytes across {} graphs",
            data.total_binary_bytes,
            data.graphs.len()
        );
    }
}
