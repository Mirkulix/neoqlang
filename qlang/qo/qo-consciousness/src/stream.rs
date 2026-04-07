use crate::state_machine::ConsciousnessState;
use tokio::sync::broadcast;

#[derive(Debug, Clone)]
pub struct ConsciousnessEvent {
    pub state: ConsciousnessState,
    pub timestamp: u64,
}

pub struct ConsciousnessStream {
    sender: broadcast::Sender<ConsciousnessEvent>,
}

impl ConsciousnessStream {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    pub fn publish(&self, state: ConsciousnessState) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let event = ConsciousnessEvent { state, timestamp };
        // Ignore send errors (no active receivers is OK)
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ConsciousnessEvent> {
        self.sender.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::ConsciousnessState;

    #[tokio::test]
    async fn publish_and_receive() {
        let stream = ConsciousnessStream::new(16);
        let mut rx = stream.subscribe();

        let state = ConsciousnessState::default();
        stream.publish(state.clone());

        let event = rx.recv().await.expect("should receive event");
        assert_eq!(event.state.heartbeat, state.heartbeat);
    }
}
