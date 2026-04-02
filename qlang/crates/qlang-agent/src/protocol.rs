//! QLANG Agent Protocol — Binary KI-to-KI communication.
//!
//! This defines how two AI agents exchange QLANG graphs:
//!
//!   Agent A ──[GraphMessage]──► Agent B
//!            ◄──[GraphMessage]──
//!
//! No JSON. No text. Binary graph exchange with typed metadata.
//! Each message is a complete, verifiable computation graph.

use serde::{Deserialize, Serialize};
use qlang_core::graph::Graph;
use qlang_core::tensor::TensorData;
use std::collections::HashMap;

/// A message exchanged between two AI agents.
///
/// This replaces text-based prompts/responses with structured graph data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMessage {
    /// Unique message identifier
    pub id: u64,
    /// Sender agent identifier
    pub from: AgentId,
    /// Receiver agent identifier
    pub to: AgentId,
    /// The computation graph (the actual "program")
    pub graph: Graph,
    /// Input data (pre-filled tensors, if any)
    pub inputs: HashMap<String, TensorData>,
    /// What the sender expects the receiver to do
    pub intent: MessageIntent,
    /// Response to a previous message (if applicable)
    pub in_reply_to: Option<u64>,
}

/// Identifies an AI agent in the protocol.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId {
    pub name: String,
    pub capabilities: Vec<Capability>,
}

/// What an agent can do.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Can execute graphs (has a runtime)
    Execute,
    /// Can compile graphs to native code (has LLVM)
    Compile,
    /// Can optimize graphs
    Optimize,
    /// Can perform IGQK compression
    Compress,
    /// Can train models (has data access)
    Train,
    /// Can verify proofs
    Verify,
}

/// What the sender wants the receiver to do with the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageIntent {
    /// "Execute this graph and return the results"
    Execute,
    /// "Optimize this graph and return the optimized version"
    Optimize,
    /// "Compress the weights in this graph using IGQK"
    Compress { method: String },
    /// "Verify the proofs in this graph"
    Verify,
    /// "Here are the results you requested"
    Result { original_message_id: u64 },
    /// "Compose this graph with yours"
    Compose,
    /// "Train this model on your data"
    Train { epochs: usize },
}

/// A conversation between agents: sequence of graph messages.
#[derive(Debug)]
pub struct AgentConversation {
    messages: Vec<GraphMessage>,
    next_id: u64,
}

impl AgentConversation {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            next_id: 0,
        }
    }

    /// Send a graph from one agent to another.
    pub fn send(
        &mut self,
        from: AgentId,
        to: AgentId,
        graph: Graph,
        inputs: HashMap<String, TensorData>,
        intent: MessageIntent,
        in_reply_to: Option<u64>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.messages.push(GraphMessage {
            id,
            from,
            to,
            graph,
            inputs,
            intent,
            in_reply_to,
        });

        id
    }

    /// Get all messages in the conversation.
    pub fn messages(&self) -> &[GraphMessage] {
        &self.messages
    }

    /// Get a specific message by ID.
    pub fn get_message(&self, id: u64) -> Option<&GraphMessage> {
        self.messages.iter().find(|m| m.id == id)
    }

    /// Serialize the entire conversation to binary.
    pub fn to_binary(&self) -> Result<Vec<u8>, serde_json::Error> {
        // Use JSON-in-binary envelope (same as graph serial format)
        let json = serde_json::to_vec(&self.messages)?;
        let mut buf = Vec::with_capacity(8 + json.len());
        buf.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x53]); // "QLMS" = QLANG Message Stream
        buf.extend_from_slice(&(self.messages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        Ok(buf)
    }
}

impl Default for AgentConversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};

    fn trainer_agent() -> AgentId {
        AgentId {
            name: "trainer".into(),
            capabilities: vec![Capability::Execute, Capability::Train],
        }
    }

    fn compressor_agent() -> AgentId {
        AgentId {
            name: "compressor".into(),
            capabilities: vec![Capability::Compress, Capability::Verify],
        }
    }

    #[test]
    fn agent_conversation() {
        let mut conv = AgentConversation::new();

        // Trainer builds a model graph and sends it to compressor
        let mut graph = Graph::new("model_weights");
        graph.add_node(
            Op::Input { name: "weights".into() },
            vec![],
            vec![TensorType::f32_matrix(128, 64)],
        );

        let mut inputs = HashMap::new();
        inputs.insert(
            "weights".into(),
            TensorData::from_f32(
                Shape::matrix(2, 2),
                &[0.5, -0.3, 0.8, -0.1],
            ),
        );

        // Message 1: Trainer → Compressor: "compress these weights"
        let msg1 = conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            inputs,
            MessageIntent::Compress { method: "ternary".into() },
            None,
        );
        assert_eq!(msg1, 0);

        // Message 2: Compressor → Trainer: "here are the compressed weights"
        let compressed_graph = Graph::new("compressed_weights");
        let msg2 = conv.send(
            compressor_agent(),
            trainer_agent(),
            compressed_graph,
            HashMap::new(),
            MessageIntent::Result { original_message_id: msg1 },
            Some(msg1),
        );
        assert_eq!(msg2, 1);

        assert_eq!(conv.messages().len(), 2);
        assert_eq!(conv.get_message(0).unwrap().from.name, "trainer");
        assert_eq!(conv.get_message(1).unwrap().in_reply_to, Some(0));
    }

    #[test]
    fn serialize_conversation() {
        let mut conv = AgentConversation::new();
        let graph = Graph::new("test");
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );

        let binary = conv.to_binary().unwrap();
        assert_eq!(&binary[0..4], &[0x51, 0x4C, 0x4D, 0x53]); // "QLMS"
    }
}
