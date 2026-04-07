use qo_llm::LlmRouter;
use crate::agent::AgentRole;
use crate::tools;

pub async fn llm_reason(
    llm: &LlmRouter,
    role: AgentRole,
    context: &str,
    task: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let messages = vec![
        ("system".to_string(), role.system_prompt().to_string()),
        ("user".to_string(), format!("Kontext: {context}\n\nAufgabe: {task}")),
    ];
    llm.chat(messages).await
}

/// Execute agent task WITH tools — not just LLM
pub async fn agent_execute_with_tools(
    llm: &LlmRouter,
    role: AgentRole,
    context: &str,
    task: &str,
    values: &qo_values::ValueScores,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    match role {
        AgentRole::Guardian => {
            // Guardian uses DETERMINISTIC value check, NOT LLM
            let check = tools::tool_values_check(task, values);
            Ok(check.output)
        }
        AgentRole::Researcher => {
            // Researcher searches the web first, then reasons with LLM
            let search_result = tools::tool_web_search(task).await;
            let enriched_context = if search_result.success {
                format!("{context}\n\nWeb-Recherche Ergebnisse:\n{}", search_result.output)
            } else {
                context.to_string()
            };
            llm_reason(llm, role, &enriched_context, task).await
        }
        AgentRole::Developer => {
            // Developer can read files and run safe commands
            let project_info = tools::tool_shell("ls -la");
            let enriched_context = format!("{context}\n\nProjekt-Verzeichnis:\n{}", project_info.output);
            llm_reason(llm, role, &enriched_context, task).await
        }
        _ => {
            // CEO, Strategist, Artisan: pure LLM reasoning
            llm_reason(llm, role, context, task).await
        }
    }
}
