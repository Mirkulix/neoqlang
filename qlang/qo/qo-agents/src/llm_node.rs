use qo_llm::LlmRouter;
use crate::agent::AgentRole;

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
