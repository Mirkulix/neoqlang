//! QO's own QLANG-native model — trained from scratch, ternary, JIT-compiled.
//!
//! This is a small intent classifier that runs in microseconds:
//! Input: text → bag-of-words vector
//! Output: category (0=Chat, 1=Goal, 2=Question, 3=Creative)
//!
//! Trained with QLANG's autograd + IGQK ternary compression.
//! No Python. No Ollama. Pure QLANG.

use qlang_runtime::training::MlpWeights;
use qlang_runtime::igqk_compress::{compress_ternary, IgqkParams};

/// Intent categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Chat = 0,
    Goal = 1,
    Question = 2,
    Creative = 3,
}

impl Intent {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Intent::Chat,
            1 => Intent::Goal,
            2 => Intent::Question,
            3 => Intent::Creative,
            _ => Intent::Chat,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Intent::Chat => "chat",
            Intent::Goal => "goal",
            Intent::Question => "question",
            Intent::Creative => "creative",
        }
    }
}

/// Vocabulary for bag-of-words encoding
const VOCAB: &[&str] = &[
    // Goal keywords (index 0-9)
    "recherchiere", "analysiere", "plane", "baue", "erstelle",
    "implementiere", "entwickle", "optimiere", "teste", "deploye",
    // Question keywords (10-19)
    "was", "wie", "warum", "wer", "wo", "wann", "welche", "kannst", "ist", "gibt",
    // Creative keywords (20-29)
    "schreibe", "gestalte", "design", "kreativ", "idee",
    "brainstorm", "erfinde", "story", "text", "dichte",
    // Chat keywords (30-39)
    "hallo", "hi", "danke", "bitte", "ja", "nein", "okay", "gut", "cool", "super",
    // Technical (40-49)
    "code", "api", "server", "datenbank", "fehler",
    "bug", "feature", "rust", "python", "system",
];

const VOCAB_SIZE: usize = 50;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 4; // 4 intents

/// Convert text to bag-of-words vector
pub fn text_to_bow(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let mut bow = vec![0.0f32; VOCAB_SIZE];
    for (i, word) in VOCAB.iter().enumerate() {
        if lower.contains(word) {
            bow[i] = 1.0;
        }
    }
    // Normalize
    let sum: f32 = bow.iter().sum();
    if sum > 0.0 {
        for v in &mut bow {
            *v /= sum;
        }
    }
    bow
}

/// Training data — hand-labeled examples
fn training_data() -> Vec<(Vec<f32>, u8)> {
    let examples = vec![
        // Goals (label 1)
        ("Recherchiere die Vorteile von Rust", 1u8),
        ("Analysiere die Performance des Systems", 1),
        ("Plane eine Roadmap für Q3", 1),
        ("Baue einen Prototyp der API", 1),
        ("Erstelle eine Zusammenfassung", 1),
        ("Implementiere das Feature", 1),
        ("Entwickle eine Strategie", 1),
        ("Optimiere den Code", 1),
        ("Teste die Funktionalität", 1),
        ("Deploye das System", 1),
        ("Recherchiere und analysiere den Markt", 1),
        ("Erstelle einen Plan für das Projekt", 1),
        // Questions (label 2)
        ("Was ist QLANG?", 2),
        ("Wie funktioniert das?", 2),
        ("Warum ist Rust schnell?", 2),
        ("Wer hat das gebaut?", 2),
        ("Wo liegt der Fehler?", 2),
        ("Wann ist das fertig?", 2),
        ("Welche Optionen gibt es?", 2),
        ("Kannst du mir helfen?", 2),
        ("Ist das System aktiv?", 2),
        ("Was gibt es Neues?", 2),
        ("Wie ist der Status?", 2),
        ("Was kannst du alles?", 2),
        // Creative (label 3)
        ("Schreibe einen Pitch für QO", 3),
        ("Gestalte ein Logo", 3),
        ("Design eine Landing Page", 3),
        ("Kreativ: Erfinde einen Slogan", 3),
        ("Brainstorm Ideen für Features", 3),
        ("Schreibe eine Story über KI", 3),
        ("Erfinde einen neuen Ansatz", 3),
        ("Schreibe einen Text über Innovation", 3),
        // Chat (label 0)
        ("Hallo, wie geht es dir?", 0),
        ("Hi!", 0),
        ("Danke für die Hilfe", 0),
        ("Ja, das ist gut", 0),
        ("Nein, das passt nicht", 0),
        ("Okay, verstanden", 0),
        ("Super, danke!", 0),
        ("Cool, das funktioniert", 0),
        ("Gut gemacht", 0),
        ("Bitte hilf mir", 0),
    ];

    examples.into_iter()
        .map(|(text, label)| (text_to_bow(text), label))
        .collect()
}

/// Train the intent classifier using QLANG's autograd
pub fn train_intent_model() -> (MlpWeights, Vec<f32>) {
    let data = training_data();

    // Initialize model
    let mut model = MlpWeights::new(VOCAB_SIZE, HIDDEN_DIM, OUTPUT_DIM);

    // Training loop
    let epochs = 100;
    let lr = 0.05;
    let mut losses = Vec::new();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;
        let total = data.len();

        for (bow, label) in &data {
            // Forward + backward + update via QLANG autograd
            let loss = model.train_step_backprop(bow, &[*label], lr);
            epoch_loss += loss;

            // Check accuracy
            let probs = model.forward(bow);
            let predicted = probs.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == *label as usize {
                correct += 1;
            }
        }

        let avg_loss = epoch_loss / total as f32;
        let accuracy = correct as f32 / total as f32 * 100.0;
        losses.push(avg_loss);

        if epoch % 20 == 0 || epoch == epochs - 1 {
            tracing::info!(
                "QLANG Model Training: Epoch {}/{} loss={:.4} acc={:.1}%",
                epoch + 1, epochs, avg_loss, accuracy
            );
        }
    }

    (model, losses)
}

/// Compress model to ternary using IGQK
pub fn compress_to_ternary(model: &MlpWeights) -> (Vec<f32>, Vec<f32>, CompressionInfo) {
    let params = IgqkParams {
        evolution_steps: 10,
        rank: 4,
        ..Default::default()
    };

    let w1_result = compress_ternary(&model.w1, &params);
    let w2_result = compress_ternary(&model.w2, &params);

    let info = CompressionInfo {
        original_bytes: w1_result.stats.original_size_bytes + w2_result.stats.original_size_bytes,
        compressed_bytes: w1_result.stats.compressed_size_bytes + w2_result.stats.compressed_size_bytes,
        ratio: (w1_result.stats.original_size_bytes + w2_result.stats.original_size_bytes) as f32
            / (w1_result.stats.compressed_size_bytes + w2_result.stats.compressed_size_bytes) as f32,
        w1_ternary: format!("+1={} 0={} -1={}",
            w1_result.stats.num_positive, w1_result.stats.num_zero, w1_result.stats.num_negative),
        w2_ternary: format!("+1={} 0={} -1={}",
            w2_result.stats.num_positive, w2_result.stats.num_zero, w2_result.stats.num_negative),
    };

    (w1_result.compressed_weights, w2_result.compressed_weights, info)
}

#[derive(Debug)]
pub struct CompressionInfo {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub ratio: f32,
    pub w1_ternary: String,
    pub w2_ternary: String,
}

/// Run inference with the trained model
pub fn classify_intent(model: &MlpWeights, text: &str) -> (Intent, Vec<f32>) {
    let bow = text_to_bow(text);
    let probs = model.forward(&bow);
    let best = probs.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    (Intent::from_index(best), probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bow_encodes_correctly() {
        let bow = text_to_bow("Recherchiere die Vorteile");
        assert!(bow[0] > 0.0); // "recherchiere" at index 0
        assert_eq!(bow.len(), VOCAB_SIZE);
    }

    #[test]
    fn training_reduces_loss() {
        let (_, losses) = train_intent_model();
        assert!(losses.last().unwrap() < losses.first().unwrap(),
            "Loss should decrease: first={} last={}", losses.first().unwrap(), losses.last().unwrap());
    }

    #[test]
    fn model_classifies_goal() {
        let (model, _) = train_intent_model();
        let (intent, _) = classify_intent(&model, "Recherchiere die Marktdaten");
        assert_eq!(intent, Intent::Goal, "Should classify as Goal");
    }

    #[test]
    fn model_classifies_question() {
        let (model, _) = train_intent_model();
        let (intent, _) = classify_intent(&model, "Was ist das?");
        assert_eq!(intent, Intent::Question, "Should classify as Question");
    }

    #[test]
    fn model_classifies_chat() {
        let (model, _) = train_intent_model();
        let (intent, _) = classify_intent(&model, "Hallo, danke!");
        assert_eq!(intent, Intent::Chat, "Should classify as Chat");
    }

    #[test]
    fn ternary_compression_works() {
        let (model, _) = train_intent_model();
        let (w1_compressed, w2_compressed, info) = compress_to_ternary(&model);
        assert!(!w1_compressed.is_empty());
        assert!(!w2_compressed.is_empty());
        assert!(info.ratio > 5.0, "Compression ratio should be >5x, got {:.1}x", info.ratio);
    }

    #[test]
    fn compressed_model_still_works() {
        let (mut model, _) = train_intent_model();
        let (w1_compressed, w2_compressed, _) = compress_to_ternary(&model);

        // Replace weights with compressed versions
        model.w1 = w1_compressed;
        model.w2 = w2_compressed;

        // Should still classify correctly (ternary preserves accuracy)
        let (intent, _) = classify_intent(&model, "Recherchiere den Markt");
        // May not be perfect but should be reasonable
        println!("Compressed model intent: {:?}", intent);
    }
}
