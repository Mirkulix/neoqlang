use crate::{
    agent::AgentRole,
    goal::{Goal, GoalStatus, SubTask},
    llm_node,
};
use qo_llm::LlmRouter;

/// Execute a goal: CEO decomposes it, then each subtask is executed by the assigned agent.
pub async fn execute_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    goal.status = GoalStatus::InProgress;

    decompose_goal(llm, goal).await?;

    let subtask_count = goal.subtasks.len();
    for i in 0..subtask_count {
        let _ = execute_subtask(llm, goal, i).await;
    }

    summarize_goal(llm, goal).await?;

    Ok(())
}

/// Step 1: CEO decomposes the goal into subtasks (populates goal.subtasks).
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

/// Step 2: Execute a single subtask by index. Returns the result string.
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

/// Step 3: CEO summarizes all subtask results. Returns the summary string.
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

/// Parse subtasks from CEO's decomposition response.
fn parse_subtasks(response: &str) -> Vec<SubTask> {
    let mut subtasks = Vec::new();
    for line in response.lines() {
        let line = line.trim();
        // Match patterns like "1. [Researcher] Do something" or "- Researcher: Do something"
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
    // Try [Role] format
    if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            let role_str = &s[1..end];
            let desc = s[end + 1..].trim().to_string();
            return (match_role(role_str), desc);
        }
    }
    // Try "Role: desc" format
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subtasks_bracket_format() {
        let input = "1. [Researcher] Analysiere den Markt\n2. [Developer] Baue den Prototyp\n3. [Strategist] Erstelle die Roadmap";
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
}
