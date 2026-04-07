use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Produce a pseudo-embedding vector for `text` with the given number of dimensions.
/// Uses a hash-based approach: different hash seeds per dimension.
/// Output is L2-normalized.
pub fn embed_text(text: &str, dimensions: usize) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }

    let mut vec: Vec<f32> = (0..dimensions)
        .map(|dim| {
            let mut hasher = DefaultHasher::new();
            dim.hash(&mut hasher);
            text.hash(&mut hasher);
            let hash_val = hasher.finish();
            // Map hash to [-1, 1]
            let normalized = (hash_val as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            normalized
        })
        .collect();

    // L2 normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_produces_correct_dimensions() {
        let v = embed_text("hello world", 128);
        assert_eq!(v.len(), 128);

        let v2 = embed_text("another text", 64);
        assert_eq!(v2.len(), 64);
    }

    #[test]
    fn embed_is_normalized() {
        let v = embed_text("normalize me", 256);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm should be 1.0, got {norm}");
    }

    #[test]
    fn similar_texts_are_closer() {
        let v_a = embed_text("the quick brown fox", 128);
        let v_b = embed_text("the quick brown fox", 128);
        let v_c = embed_text("completely unrelated xyz 12345", 128);

        let dist_same = crate::hnsw::cosine_distance(&v_a, &v_b);
        let dist_diff = crate::hnsw::cosine_distance(&v_a, &v_c);

        // Same text must have distance 0
        assert!(dist_same < 1e-5, "same text distance should be 0, got {dist_same}");
        // Different text should be farther away
        assert!(
            dist_diff > dist_same,
            "different text ({dist_diff}) should be farther than same text ({dist_same})"
        );
    }
}
