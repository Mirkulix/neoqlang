use crate::{
    cloud::{CloudClient, CloudMessage},
    groq::{GroqClient, GroqMessage},
};
use std::error::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Local,
    Groq,
    Cloud,
}

pub struct LlmRouter {
    groq: Option<GroqClient>,
    cloud: Option<CloudClient>,
}

impl LlmRouter {
    pub fn new(
        groq_api_key: Option<String>,
        cloud_config: Option<(String, String, String)>,
    ) -> Self {
        Self {
            groq: groq_api_key.map(GroqClient::new),
            cloud: cloud_config
                .map(|(api_key, base_url, model)| CloudClient::new(api_key, base_url, model)),
        }
    }

    /// Heuristic complexity score in [0.0, 1.0].
    /// Considers prompt length and presence of complexity keywords.
    pub fn score_complexity(&self, prompt: &str) -> f32 {
        let len_score = (prompt.len() as f32 / 2000.0).min(1.0) * 0.5;

        let complex_keywords = [
            "architecture",
            "design",
            "optimize",
            "refactor",
            "security",
            "algorithm",
            "implement",
            "analyze",
            "complex",
            "system",
        ];
        let lower = prompt.to_lowercase();
        let keyword_hits = complex_keywords
            .iter()
            .filter(|kw| lower.contains(*kw))
            .count();
        let keyword_score = (keyword_hits as f32 / complex_keywords.len() as f32) * 0.5;

        (len_score + keyword_score).min(1.0)
    }

    /// Select a tier based on complexity score.
    pub fn select_tier(&self, complexity: f32) -> Tier {
        if complexity < 0.3 {
            Tier::Local
        } else if complexity < 0.7 {
            Tier::Groq
        } else {
            Tier::Cloud
        }
    }

    /// Route the conversation to the appropriate tier.
    /// In Phase 1, Local falls back to Groq when no local model is available.
    pub async fn chat(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let prompt = messages
            .last()
            .map(|(_, c)| c.as_str())
            .unwrap_or_default();
        let complexity = self.score_complexity(prompt);
        let tier = self.select_tier(complexity);

        tracing::debug!(?tier, complexity, "routing request");

        match tier {
            Tier::Cloud => {
                if let Some(cloud) = &self.cloud {
                    let msgs: Vec<CloudMessage> = messages
                        .into_iter()
                        .map(|(role, content)| CloudMessage { role, content })
                        .collect();
                    return cloud.chat(msgs).await;
                }
                // fall through to Groq
                self.groq_chat(messages).await
            }
            Tier::Groq | Tier::Local => {
                // Phase 1: Local falls back to Groq
                self.groq_chat(messages).await
            }
        }
    }

    async fn groq_chat(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        if let Some(groq) = &self.groq {
            let msgs: Vec<GroqMessage> = messages
                .into_iter()
                .map(|(role, content)| GroqMessage { role, content })
                .collect();
            groq.chat(msgs).await
        } else {
            Err("No LLM backend configured".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> LlmRouter {
        LlmRouter::new(None, None)
    }

    #[test]
    fn simple_prompt_routes_to_local() {
        let r = router();
        let complexity = r.score_complexity("hi");
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Local);
    }

    #[test]
    fn medium_prompt_routes_to_groq() {
        let r = router();
        // 1600 chars → len_score = (1600/2000)*0.5 = 0.4, no keywords → total = 0.4 (Groq range)
        let prompt = "a".repeat(1600);
        let complexity = r.score_complexity(&prompt);
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Groq);
    }

    #[test]
    fn complex_prompt_routes_to_cloud() {
        let r = router();
        // Long prompt with many complexity keywords
        let prompt = format!(
            "{} architecture design optimize refactor security algorithm implement analyze complex system",
            "a".repeat(1500)
        );
        let complexity = r.score_complexity(&prompt);
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Cloud);
    }
}
