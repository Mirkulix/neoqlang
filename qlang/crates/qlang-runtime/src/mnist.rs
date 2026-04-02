//! MNIST Data Loader — Load the real MNIST dataset.
//!
//! Downloads and parses the IDX binary format:
//! - train-images-idx3-ubyte (60000 images, 28×28)
//! - train-labels-idx1-ubyte (60000 labels, 0-9)
//! - t10k-images-idx3-ubyte  (10000 test images)
//! - t10k-labels-idx1-ubyte  (10000 test labels)
//!
//! If files don't exist, generates synthetic MNIST-like data.
//!
//! ## Downloading real MNIST data
//!
//! To use real MNIST data, download the following files into your data directory:
//!
//! ```text
//! https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
//! https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
//! https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
//! https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
//! ```
//!
//! Then decompress with `gunzip *.gz` and pass the directory to `MnistData::load()`.

use std::fmt;
use std::fs;
use std::path::Path;

/// Errors that can occur during MNIST loading.
#[derive(Debug)]
pub enum MnistError {
    /// IDX file has invalid magic number.
    InvalidMagic { expected: u32, got: u32 },
    /// IO error reading file.
    Io(std::io::Error),
    /// Files not found and user must download manually.
    FilesNotFound {
        data_dir: String,
        missing: Vec<String>,
    },
}

impl fmt::Display for MnistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MnistError::InvalidMagic { expected, got } => {
                write!(f, "Invalid IDX magic number: expected {expected}, got {got}")
            }
            MnistError::Io(e) => write!(f, "IO error: {e}"),
            MnistError::FilesNotFound { data_dir, missing } => {
                writeln!(f, "MNIST files not found in '{data_dir}'.")?;
                writeln!(f, "Missing files: {}", missing.join(", "))?;
                writeln!(f)?;
                writeln!(f, "Please download manually from:")?;
                for name in missing {
                    writeln!(
                        f,
                        "  https://storage.googleapis.com/cvdf-datasets/mnist/{name}.gz"
                    )?;
                }
                writeln!(f)?;
                write!(f, "Then decompress with: gunzip *.gz")
            }
        }
    }
}

impl std::error::Error for MnistError {}

impl From<std::io::Error> for MnistError {
    fn from(e: std::io::Error) -> Self {
        MnistError::Io(e)
    }
}

const MNIST_FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

/// MNIST dataset.
pub struct MnistData {
    pub train_images: Vec<f32>, // [n_train, 784], normalized to [0, 1]
    pub train_labels: Vec<u8>,  // [n_train], values 0-9
    pub test_images: Vec<f32>,  // [n_test, 784]
    pub test_labels: Vec<u8>,   // [n_test]
    pub n_train: usize,
    pub n_test: usize,
    pub image_size: usize, // 784 = 28×28
    pub n_classes: usize,  // 10
}

/// Download instructions for MNIST data.
///
/// Because we avoid external HTTP dependencies, this function checks whether the
/// MNIST IDX files exist in `data_dir` and returns an error with download
/// instructions if any are missing.
pub fn download_mnist(data_dir: &str) -> Result<(), MnistError> {
    let dir = Path::new(data_dir);
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }

    let missing: Vec<String> = MNIST_FILES
        .iter()
        .filter(|name| !dir.join(name).exists())
        .map(|s| s.to_string())
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(MnistError::FilesNotFound {
            data_dir: data_dir.to_string(),
            missing,
        })
    }
}

impl MnistData {
    /// Try to load from IDX files, or generate synthetic data.
    pub fn load(data_dir: &str) -> Self {
        let train_images_path = format!("{}/train-images-idx3-ubyte", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
        let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

        if Path::new(&train_images_path).exists() {
            // Load real MNIST
            let train_images = parse_idx_images(&train_images_path);
            let train_labels = parse_idx_labels(&train_labels_path);
            let test_images = parse_idx_images(&test_images_path);
            let test_labels = parse_idx_labels(&test_labels_path);

            let n_train = train_labels.len();
            let n_test = test_labels.len();

            Self {
                train_images,
                train_labels,
                test_images,
                test_labels,
                n_train,
                n_test,
                image_size: 784,
                n_classes: 10,
            }
        } else {
            // Generate synthetic MNIST-like data
            Self::synthetic(1000, 200)
        }
    }

    /// Generate synthetic MNIST-like data with realistic variation.
    ///
    /// Each digit class has a distinct structural pattern. Variation is added
    /// through random translation offsets, stroke thickness jitter, and
    /// per-pixel noise so that no two samples are identical.
    pub fn synthetic(n_train: usize, n_test: usize) -> Self {
        let image_size = 784; // 28×28

        let mut train_images = vec![0.0f32; n_train * image_size];
        let mut train_labels = vec![0u8; n_train];
        let mut test_images = vec![0.0f32; n_test * image_size];
        let mut test_labels = vec![0u8; n_test];

        for i in 0..n_train {
            let label = (i % 10) as u8;
            train_labels[i] = label;
            draw_digit(
                &mut train_images[i * image_size..(i + 1) * image_size],
                label,
                i,
            );
        }

        for i in 0..n_test {
            let label = (i % 10) as u8;
            test_labels[i] = label;
            draw_digit(
                &mut test_images[i * image_size..(i + 1) * image_size],
                label,
                i + n_train,
            );
        }

        Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            n_train,
            n_test,
            image_size,
            n_classes: 10,
        }
    }

    /// Get a batch of training data.
    pub fn train_batch(&self, offset: usize, batch_size: usize) -> (&[f32], &[u8]) {
        let start = offset % self.n_train;
        let end = (start + batch_size).min(self.n_train);
        (
            &self.train_images[start * self.image_size..end * self.image_size],
            &self.train_labels[start..end],
        )
    }

    /// Return a human-readable summary of dataset statistics.
    pub fn summary(&self) -> String {
        let train_nonzero: usize = self.train_images.iter().filter(|&&x| x > 0.0).count();
        let test_nonzero: usize = self.test_images.iter().filter(|&&x| x > 0.0).count();
        let train_density = if self.n_train > 0 {
            train_nonzero as f64 / (self.n_train * self.image_size) as f64
        } else {
            0.0
        };
        let test_density = if self.n_test > 0 {
            test_nonzero as f64 / (self.n_test * self.image_size) as f64
        } else {
            0.0
        };

        let dist = self.class_distribution();
        let mut dist_str = String::new();
        for (label, count) in &dist {
            dist_str.push_str(&format!("  {}: {}\n", label, count));
        }

        format!(
            "MNIST Dataset Summary\n\
             =====================\n\
             Training samples: {}\n\
             Test samples:     {}\n\
             Image size:       {}x{} ({})\n\
             Classes:          {}\n\
             Train pixel density: {:.3}\n\
             Test pixel density:  {:.3}\n\
             \n\
             Class distribution (train):\n\
             {}",
            self.n_train,
            self.n_test,
            28,
            28,
            self.image_size,
            self.n_classes,
            train_density,
            test_density,
            dist_str,
        )
    }

    /// Render a sample image as ASCII art (28x28, using `#` for lit pixels and
    /// space for dark pixels).
    ///
    /// Uses the training set. Returns an empty string if `index` is out of range.
    pub fn visualize_sample(&self, index: usize) -> String {
        if index >= self.n_train {
            return String::new();
        }
        let start = index * self.image_size;
        let pixels = &self.train_images[start..start + self.image_size];
        let label = self.train_labels[index];

        let mut out = format!("Label: {}\n", label);
        for row in 0..28 {
            for col in 0..28 {
                let v = pixels[row * 28 + col];
                if v > 0.75 {
                    out.push('#');
                } else if v > 0.4 {
                    out.push('+');
                } else if v > 0.15 {
                    out.push('.');
                } else {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
        out
    }

    /// Count the number of training samples per class label.
    ///
    /// Returns a vector of `(label, count)` pairs sorted by label 0..9.
    pub fn class_distribution(&self) -> Vec<(u8, usize)> {
        let mut counts = [0usize; 10];
        for &label in &self.train_labels {
            if (label as usize) < 10 {
                counts[label as usize] += 1;
            }
        }
        (0..10).map(|i| (i as u8, counts[i])).collect()
    }
}

// ---------------------------------------------------------------------------
// Simple deterministic PRNG (xorshift32) for reproducible noise
// ---------------------------------------------------------------------------

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Draw a synthetic digit pattern on a 28×28 image with realistic variation.
///
/// Improvements over the basic version:
/// - Per-sample translation offset (random-ish via seed)
/// - Stroke thickness variation (draw neighbouring pixels with lower intensity)
/// - Per-pixel noise (random flips / intensity jitter)
/// - More structurally distinct patterns per class
fn draw_digit(image: &mut [f32], digit: u8, seed: usize) {
    let w = 28usize;

    // Deterministic pseudo-random state derived from seed
    let mut rng = (seed as u32).wrapping_mul(2654435761).wrapping_add(1);

    // Random offsets for translation variation (-2..+2)
    let offset_x = ((xorshift32(&mut rng) % 5) as i32) - 2;
    let offset_y = ((xorshift32(&mut rng) % 5) as i32) - 2;

    // Stroke thickness multiplier (0.7 .. 1.0)
    let thickness = 0.7 + (xorshift32(&mut rng) % 300) as f32 / 1000.0;

    // Safe pixel setter with anti-aliased thickness
    let set_thick = |img: &mut [f32], x: i32, y: i32, val: f32| {
        // Centre pixel
        if x >= 0 && x < w as i32 && y >= 0 && y < w as i32 {
            let idx = y as usize * w + x as usize;
            img[idx] = img[idx].max(val);
        }
        // Neighbouring pixels for thickness
        let neigh_val = val * thickness * 0.5;
        for &(dx, dy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < w as i32 {
                let idx = ny as usize * w + nx as usize;
                img[idx] = img[idx].max(neigh_val);
            }
        }
    };

    let ox = offset_x;
    let oy = offset_y;

    match digit {
        0 => {
            // Ellipse (oval)
            for angle in 0..80 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 80.0;
                let x = 14 + ox + (a.cos() * 7.0) as i32;
                let y = 14 + oy + (a.sin() * 9.0) as i32;
                set_thick(image, x, y, 1.0);
            }
        }
        1 => {
            // Vertical line with small serif at top and base
            for y in 5..23 {
                set_thick(image, 14 + ox, y + oy, 1.0);
            }
            // Top serif
            set_thick(image, 13 + ox, 6 + oy, 0.8);
            set_thick(image, 12 + ox, 7 + oy, 0.6);
            // Base
            for x in 12..17 {
                set_thick(image, x + ox, 23 + oy, 0.9);
            }
        }
        2 => {
            // Top arc + diagonal + bottom line
            for angle in 0..30 {
                let a = angle as f32 * std::f32::consts::PI / 30.0;
                let x = 14 + ox + (a.cos() * 6.0) as i32;
                let y = 9 + oy - (a.sin() * 4.0) as i32;
                set_thick(image, x, y, 1.0);
            }
            for i in 0..12 {
                let x = 20 + ox - i;
                let y = 9 + oy + i;
                set_thick(image, x, y, 1.0);
            }
            for x in 8..21 {
                set_thick(image, x + ox, 21 + oy, 1.0);
            }
        }
        3 => {
            // Three horizontal bars + right vertical
            for x in 10..20 {
                set_thick(image, x + ox, 6 + oy, 1.0);
                set_thick(image, x + ox, 13 + oy, 1.0);
                set_thick(image, x + ox, 21 + oy, 1.0);
            }
            for y in 6..22 {
                set_thick(image, 19 + ox, y + oy, 1.0);
            }
        }
        4 => {
            // Left vertical (top half) + horizontal bar + right vertical (full)
            for y in 4..14 {
                set_thick(image, 10 + ox, y + oy, 1.0);
            }
            for x in 10..21 {
                set_thick(image, x + ox, 14 + oy, 1.0);
            }
            for y in 4..24 {
                set_thick(image, 18 + ox, y + oy, 1.0);
            }
        }
        5 => {
            // Top bar, left vertical (top half), middle bar, right vertical (bottom half), bottom bar
            for x in 8..20 {
                set_thick(image, x + ox, 5 + oy, 1.0);
            }
            for y in 5..13 {
                set_thick(image, 8 + ox, y + oy, 1.0);
            }
            for x in 8..20 {
                set_thick(image, x + ox, 13 + oy, 1.0);
            }
            for y in 13..22 {
                set_thick(image, 19 + ox, y + oy, 1.0);
            }
            for x in 8..20 {
                set_thick(image, x + ox, 21 + oy, 1.0);
            }
        }
        6 => {
            // Left vertical full + bottom circle
            for y in 4..23 {
                set_thick(image, 10 + ox, y + oy, 1.0);
            }
            for angle in 0..50 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 50.0;
                let x = 15 + ox + (a.cos() * 5.0) as i32;
                let y = 17 + oy + (a.sin() * 5.0) as i32;
                set_thick(image, x, y, 1.0);
            }
        }
        7 => {
            // Top bar + diagonal going down-left
            for x in 8..22 {
                set_thick(image, x + ox, 5 + oy, 1.0);
            }
            for i in 0..18 {
                let x = 21 + ox - (i * 2 / 3);
                let y = 5 + oy + i;
                set_thick(image, x, y, 1.0);
            }
        }
        8 => {
            // Two stacked circles
            for angle in 0..50 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 50.0;
                let x_top = 14 + ox + (a.cos() * 5.0) as i32;
                let y_top = 9 + oy + (a.sin() * 4.0) as i32;
                set_thick(image, x_top, y_top, 1.0);
                let x_bot = 14 + ox + (a.cos() * 5.0) as i32;
                let y_bot = 19 + oy + (a.sin() * 4.0) as i32;
                set_thick(image, x_bot, y_bot, 1.0);
            }
        }
        9 => {
            // Top circle + right vertical
            for angle in 0..50 {
                let a = angle as f32 * std::f32::consts::PI * 2.0 / 50.0;
                let x = 14 + ox + (a.cos() * 5.0) as i32;
                let y = 10 + oy + (a.sin() * 5.0) as i32;
                set_thick(image, x, y, 1.0);
            }
            for y in 10..24 {
                set_thick(image, 19 + ox, y + oy, 1.0);
            }
        }
        _ => {}
    }

    // Add per-pixel noise: randomly flip some pixels and add intensity jitter
    for i in 0..image.len() {
        let r = xorshift32(&mut rng);
        // ~3% chance to flip a dark pixel to faint
        if image[i] < 0.1 && r % 33 == 0 {
            image[i] = 0.15 + (r % 100) as f32 / 500.0;
        }
        // Add slight intensity jitter to lit pixels
        if image[i] > 0.3 {
            let jitter = ((r >> 8) % 100) as f32 / 500.0 - 0.1; // -0.1 .. +0.1
            image[i] = (image[i] + jitter).clamp(0.0, 1.0);
        }
    }
}

/// Parse IDX image file format.
fn parse_idx_images(path: &str) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(magic, 2051, "Invalid image magic number");

    let _n_images = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let _n_rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let _n_cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let pixels = &data[16..];
    pixels.iter().map(|&b| b as f32 / 255.0).collect()
}

/// Parse IDX label file format.
fn parse_idx_labels(path: &str) -> Vec<u8> {
    let data = fs::read(path).unwrap();
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(magic, 2049, "Invalid label magic number");

    data[8..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_data_has_correct_sizes() {
        let data = MnistData::synthetic(100, 20);
        assert_eq!(data.n_train, 100);
        assert_eq!(data.n_test, 20);
        assert_eq!(data.train_images.len(), 100 * 784);
        assert_eq!(data.train_labels.len(), 100);
        assert_eq!(data.test_images.len(), 20 * 784);
        assert_eq!(data.test_labels.len(), 20);
        assert_eq!(data.image_size, 784);
        assert_eq!(data.n_classes, 10);
        assert!(data.train_labels.iter().all(|&l| l < 10));
        assert!(data.test_labels.iter().all(|&l| l < 10));
    }

    #[test]
    fn all_ten_classes_present() {
        let data = MnistData::synthetic(100, 20);
        let dist = data.class_distribution();
        assert_eq!(dist.len(), 10);
        for (label, count) in &dist {
            assert!(*label < 10, "unexpected label {}", label);
            assert!(*count > 0, "class {} has zero samples", label);
        }
    }

    #[test]
    fn patterns_distinct_between_classes() {
        // Each digit should produce a different pattern
        let mut patterns: Vec<Vec<u8>> = Vec::new();
        for d in 0..10u8 {
            let mut img = vec![0.0f32; 784];
            draw_digit(&mut img, d, 42); // same seed so only digit differs
            let binary: Vec<u8> = img.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
            patterns.push(binary);
        }

        for i in 0..10 {
            for j in (i + 1)..10 {
                assert_ne!(
                    patterns[i], patterns[j],
                    "Digit {i} and {j} have identical patterns"
                );
            }
        }

        // Additionally verify that patterns differ substantially (>5% of pixels)
        for i in 0..10 {
            for j in (i + 1)..10 {
                let diff: usize = patterns[i]
                    .iter()
                    .zip(patterns[j].iter())
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(
                    diff > 784 / 20,
                    "Digits {} and {} differ by only {} pixels",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn visualization_produces_nonempty_string() {
        let data = MnistData::synthetic(20, 5);
        let viz = data.visualize_sample(0);
        assert!(!viz.is_empty(), "visualization should be non-empty");
        assert!(viz.contains("Label:"), "should contain label header");
        // Should have 28 rows of content (plus label line)
        let lines: Vec<&str> = viz.lines().collect();
        assert_eq!(lines.len(), 29, "expected 1 label line + 28 image rows");
        // At least some lit pixels
        assert!(
            viz.contains('#') || viz.contains('+'),
            "visualization should contain lit pixels"
        );

        // Out of range returns empty
        let empty = data.visualize_sample(9999);
        assert!(empty.is_empty());
    }

    #[test]
    fn class_distribution_is_balanced() {
        let data = MnistData::synthetic(1000, 200);
        let dist = data.class_distribution();
        assert_eq!(dist.len(), 10);

        // With n_train=1000 and label = i%10, each class should have exactly 100
        for (label, count) in &dist {
            assert_eq!(
                *count, 100,
                "class {} should have 100 samples but has {}",
                label, count
            );
        }
    }

    #[test]
    fn batch_loading() {
        let data = MnistData::synthetic(100, 20);
        let (images, labels) = data.train_batch(0, 10);
        assert_eq!(images.len(), 10 * 784);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn summary_contains_key_info() {
        let data = MnistData::synthetic(100, 20);
        let s = data.summary();
        assert!(s.contains("100"), "summary should mention train count");
        assert!(s.contains("20"), "summary should mention test count");
        assert!(s.contains("28"), "summary should mention image dimensions");
        assert!(
            s.contains("Class distribution"),
            "summary should show distribution"
        );
    }

    #[test]
    fn download_mnist_reports_missing_files() {
        // Use a temp dir that won't have MNIST files
        let dir = "/tmp/qlang_test_mnist_missing";
        let _ = fs::remove_dir_all(dir);
        let result = download_mnist(dir);
        assert!(result.is_err());
        match result.unwrap_err() {
            MnistError::FilesNotFound { missing, .. } => {
                assert_eq!(missing.len(), 4);
            }
            other => panic!("Expected FilesNotFound, got: {other}"),
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn synthetic_images_have_variation() {
        // Two samples of the same digit with different seeds should differ
        let mut img1 = vec![0.0f32; 784];
        let mut img2 = vec![0.0f32; 784];
        draw_digit(&mut img1, 3, 0);
        draw_digit(&mut img2, 3, 50);

        let diff: usize = img1
            .iter()
            .zip(img2.iter())
            .filter(|(a, b)| ((*a) - (*b)).abs() > 0.05)
            .count();
        assert!(
            diff > 10,
            "same digit with different seeds should vary, but diff={}",
            diff
        );
    }
}
