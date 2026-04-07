use crate::{
    cloud::{CloudClient, CloudMessage},
    groq::{GroqClient, GroqMessage},
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Local,
    Groq,
    Cloud,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStats {
    pub name: String,
    pub model: String,
    pub requests: u64,
    pub total_tokens_estimate: u64,
    pub cost_usd: f64,
    pub avg_latency_ms: u64,
    pub status: String,
}

#[derive(Debug, Default)]
pub struct CostTracker {
    pub groq_requests: AtomicU64,
    pub groq_tokens: AtomicU64,
    pub cloud_requests: AtomicU64,
    pub cloud_tokens: AtomicU64,
    pub local_requests: AtomicU64,
    pub total_latency_groq_ms: AtomicU64,
    pub total_latency_cloud_ms: AtomicU64,
}

pub struct LlmRouter {
    groq: Option<GroqClient>,
    cloud: Option<CloudClient>,
    cloud_model: Option<String>,
    cost_tracker: Arc<CostTracker>,
}

impl LlmRouter {
    pub fn new(
        groq_api_key: Option<String>,
        cloud_config: Option<(String, String, String)>,
    ) -> Self {
        let cloud_model = cloud_config.as_ref().map(|(_, _, model)| model.clone());
        Self {
            groq: groq_api_key.map(GroqClient::new),
            cloud: cloud_config
                .map(|(api_key, base_url, model)| CloudClient::new(api_key, base_url, model)),
            cloud_model,
            cost_tracker: Arc::new(CostTracker::default()),
        }
    }

    /// Expose cost tracker (e.g. for sharing with server state).
    pub fn cost_tracker(&self) -> Arc<CostTracker> {
        self.cost_tracker.clone()
    }

    /// Returns stats for each configured provider.
    pub fn provider_stats(&self) -> Vec<ProviderStats> {
        let mut stats = Vec::new();

        // Groq
        let groq_requests = self.cost_tracker.groq_requests.load(Ordering::Relaxed);
        let groq_tokens = self.cost_tracker.groq_tokens.load(Ordering::Relaxed);
        let total_latency_groq = self.cost_tracker.total_latency_groq_ms.load(Ordering::Relaxed);
        let avg_latency_groq = if groq_requests > 0 {
            total_latency_groq / groq_requests
        } else {
            0
        };
        let groq_status = if self.groq.is_some() { "active" } else { "inactive" }.to_string();
        stats.push(ProviderStats {
            name: "Groq".to_string(),
            model: "llama-3.3-70b-versatile".to_string(),
            requests: groq_requests,
            total_tokens_estimate: groq_tokens,
            cost_usd: 0.0,
            avg_latency_ms: avg_latency_groq,
            status: groq_status,
        });

        // Cloud
        let cloud_requests = self.cost_tracker.cloud_requests.load(Ordering::Relaxed);
        let cloud_tokens = self.cost_tracker.cloud_tokens.load(Ordering::Relaxed);
        let total_latency_cloud = self.cost_tracker.total_latency_cloud_ms.load(Ordering::Relaxed);
        let avg_latency_cloud = if cloud_requests > 0 {
            total_latency_cloud / cloud_requests
        } else {
            0
        };
        // Estimate cost at $0.01/1K tokens average
        let cloud_cost = (cloud_tokens as f64 / 1000.0) * 0.01;
        let cloud_status = if self.cloud.is_some() { "active" } else { "inactive" }.to_string();
        let cloud_model = self.cloud_model.clone().unwrap_or_default();
        stats.push(ProviderStats {
            name: "Cloud".to_string(),
            model: cloud_model,
            requests: cloud_requests,
            total_tokens_estimate: cloud_tokens,
            cost_usd: cloud_cost,
            avg_latency_ms: avg_latency_cloud,
            status: cloud_status,
        });

        // Local IGQK
        let local_requests = self.cost_tracker.local_requests.load(Ordering::Relaxed);
        stats.push(ProviderStats {
            name: "QO-LLM (IGQK)".to_string(),
            model: "qwen2.5-0.5b-ternary".to_string(),
            requests: local_requests,
            total_tokens_estimate: 0,
            cost_usd: 0.0,
            avg_latency_ms: 0,
            status: "coming_soon".to_string(),
        });

        stats
    }

    /// Returns total estimated cost across all providers.
    pub fn total_cost(&self) -> f64 {
        let cloud_tokens = self.cost_tracker.cloud_tokens.load(Ordering::Relaxed);
        (cloud_tokens as f64 / 1000.0) * 0.01
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
                    let start = Instant::now();
                    let result = cloud.chat(msgs).await?;
                    let elapsed = start.elapsed().as_millis() as u64;
                    let tokens = (result.len() / 4) as u64;
                    self.cost_tracker.cloud_requests.fetch_add(1, Ordering::Relaxed);
                    self.cost_tracker.cloud_tokens.fetch_add(tokens, Ordering::Relaxed);
                    self.cost_tracker.total_latency_cloud_ms.fetch_add(elapsed, Ordering::Relaxed);
                    return Ok(result);
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
            let start = Instant::now();
            let result = groq.chat(msgs).await?;
            let elapsed = start.elapsed().as_millis() as u64;
            let tokens = (result.len() / 4) as u64;
            self.cost_tracker.groq_requests.fetch_add(1, Ordering::Relaxed);
            self.cost_tracker.groq_tokens.fetch_add(tokens, Ordering::Relaxed);
            self.cost_tracker.total_latency_groq_ms.fetch_add(elapsed, Ordering::Relaxed);
            Ok(result)
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

    #[test]
    fn provider_stats_returns_three_providers() {
        let r = router();
        let stats = r.provider_stats();
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].name, "Groq");
        assert_eq!(stats[1].name, "Cloud");
        assert_eq!(stats[2].name, "QO-LLM (IGQK)");
    }

    #[test]
    fn total_cost_starts_at_zero() {
        let r = router();
        assert_eq!(r.total_cost(), 0.0);
    }

    #[test]
    fn groq_status_inactive_when_no_key() {
        let r = router();
        let stats = r.provider_stats();
        assert_eq!(stats[0].status, "inactive");
    }
}
