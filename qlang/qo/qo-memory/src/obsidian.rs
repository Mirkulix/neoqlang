use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct ObsidianBridge {
    pub vault_path: PathBuf,
}

impl ObsidianBridge {
    pub fn new(vault_path: impl Into<PathBuf>) -> Self {
        Self {
            vault_path: vault_path.into(),
        }
    }

    pub async fn write_consciousness_log(&self, date: &str, entry: &str) -> std::io::Result<()> {
        let dir = self.vault_path.join("consciousness");
        fs::create_dir_all(&dir).await?;
        let file_path = dir.join(format!("{date}.md"));
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await?;
        let content = format!(
            "---\ndate: {date}\ntype: consciousness-log\n---\n\n{entry}\n"
        );
        file.write_all(content.as_bytes()).await?;
        Ok(())
    }

    pub async fn write_chat_log(
        &self,
        date: &str,
        role: &str,
        content: &str,
    ) -> std::io::Result<()> {
        let dir = self.vault_path.join("chats");
        fs::create_dir_all(&dir).await?;
        let file_path = dir.join(format!("{date}.md"));
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await?;
        let entry = format!(
            "---\ndate: {date}\ntype: chat-log\n---\n\n**{role}**: {content}\n"
        );
        file.write_all(entry.as_bytes()).await?;
        Ok(())
    }

    pub async fn list_sections(&self) -> std::io::Result<Vec<String>> {
        let mut sections = Vec::new();
        let mut read_dir = fs::read_dir(&self.vault_path).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    sections.push(name.to_owned());
                }
            }
        }
        sections.sort();
        Ok(sections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn write_and_read_consciousness() {
        let dir = tempdir().unwrap();
        let bridge = ObsidianBridge::new(dir.path());

        bridge
            .write_consciousness_log("2026-04-07", "Today I thought about embeddings.")
            .await
            .unwrap();

        let file_path = dir.path().join("consciousness").join("2026-04-07.md");
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert!(content.contains("consciousness-log"));
        assert!(content.contains("Today I thought about embeddings."));
    }

    #[tokio::test]
    async fn list_sections() {
        let dir = tempdir().unwrap();
        let bridge = ObsidianBridge::new(dir.path());

        bridge
            .write_consciousness_log("2026-04-07", "entry one")
            .await
            .unwrap();
        bridge
            .write_chat_log("2026-04-07", "user", "hello")
            .await
            .unwrap();

        let sections = bridge.list_sections().await.unwrap();
        assert!(sections.contains(&"consciousness".to_string()));
        assert!(sections.contains(&"chats".to_string()));
    }
}
