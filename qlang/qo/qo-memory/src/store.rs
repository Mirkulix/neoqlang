use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::Arc;

const CHAT_TABLE: TableDefinition<u64, &str> = TableDefinition::new("chat");
const KV_TABLE: TableDefinition<&str, &str> = TableDefinition::new("kv");
const HISTORY_TABLE: TableDefinition<u64, &str> = TableDefinition::new("action_history");

pub struct Store {
    db: Arc<Database>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, redb::Error> {
        let db = Arc::new(Database::create(path)?);
        // Ensure tables exist
        let write_txn = db.begin_write()?;
        {
            write_txn.open_table(CHAT_TABLE)?;
            write_txn.open_table(KV_TABLE)?;
            write_txn.open_table(HISTORY_TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    /// Expose the underlying database for sharing with other stores (e.g. GraphStore)
    pub fn db(&self) -> Arc<Database> {
        self.db.clone()
    }

    pub fn log_action(
        &self,
        action_type: &str,
        description: &str,
        details: &str,
    ) -> Result<(), redb::Error> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let entry = format!(
            r#"{{"id":{},"timestamp":{},"action_type":"{}","description":"{}","details":"{}"}}"#,
            id,
            timestamp,
            action_type,
            description.replace('"', "\\\"").replace('\n', "\\n"),
            details.replace('"', "\\\"").replace('\n', "\\n"),
        );
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(HISTORY_TABLE)?;
            table.insert(id, entry.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get_history(&self, limit: usize) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(HISTORY_TABLE)?;
        let mut results = Vec::new();
        // Collect all, then take last N (most recent = highest id)
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
        }
        // Return last `limit` entries (most recent)
        let start = if results.len() > limit { results.len() - limit } else { 0 };
        let mut slice = results[start..].to_vec();
        slice.reverse();
        Ok(slice)
    }

    pub fn store_chat(&self, id: u64, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CHAT_TABLE)?;
            table.insert(id, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn chat_history(&self, limit: usize) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHAT_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    pub fn set(&self, key: &str, value: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(KV_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Option<String>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(KV_TABLE)?;
        Ok(table.get(key)?.map(|v| v.value().to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn store_and_retrieve_chat() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let store = Store::open(&db_path).unwrap();

        store.store_chat(1, r#"{"role":"user","content":"hello"}"#).unwrap();
        store.store_chat(2, r#"{"role":"assistant","content":"hi"}"#).unwrap();

        let history = store.chat_history(10).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].0, 1);
        assert!(history[0].1.contains("user"));
        assert_eq!(history[1].0, 2);
        assert!(history[1].1.contains("assistant"));
    }

    #[test]
    fn kv_set_get() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("kv_test.redb");
        let store = Store::open(&db_path).unwrap();

        store.set("model", "qo-1").unwrap();
        store.set("version", "0.1.0").unwrap();

        assert_eq!(store.get("model").unwrap(), Some("qo-1".to_string()));
        assert_eq!(store.get("version").unwrap(), Some("0.1.0".to_string()));
        assert_eq!(store.get("missing").unwrap(), None);
    }
}
