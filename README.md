# IGQK - Information-Geometric Quantum Compression

**Eine theoretische Grundlage für neuronale Netzwerk-Kompression, die Information Geometry, Quantenmechanik und Kompressionstheorie vereint.**

## 📖 Übersicht

IGQK ist ein mathematisch rigoroses Framework für die Kompression neuronaler Netze, das drei Schlüsselkonzepte kombiniert:

1. **Informationsgeometrie**: Statistische Mannigfaltigkeiten mit Fisher-Informationsmetrik
2. **Quantenmechanik**: Superposition und Verschränkung für Gewichtsoptimierung
3. **Kompressionstheorie**: Projektion auf niedrig-dimensionale Untermannigfaltigkeiten

### Kernidee

Statt Gewichte direkt zu optimieren, behandelt IGQK sie als **Quantenzustände** auf einer **statistischen Mannigfaltigkeit**. Der Trainingsalgorithmus verwendet **Quantengradientenfluss**, der sowohl Exploration (unitäre Evolution) als auch Konvergenz (dissipative Evolution) kombiniert.

### Mathematische Grundlagen

Die zentrale Evolutionsgleichung (Definition 2.1):

```
dρ/dt = -i[H, ρ] - γ{G⁻¹∇L, ρ}
```

Dabei:
- `ρ`: Dichtematrix (Quantenzustand der Gewichte)
- `H = -Δ_M`: Laplace-Beltrami-Operator (Quantenexploration)
- `[H, ρ]`: Kommutator (unitäre Evolution)
- `{G⁻¹∇L, ρ}`: Antikommutator (dissipativer Gradientenabstieg)
- `G`: Fisher-Informationsmetrik
- `γ`: Dämpfungsparameter

## 🎯 Hauptmerkmale

- ✅ **Theoretisch fundiert**: Basiert auf Riemannscher Geometrie und Quanteninformationstheorie
- ✅ **Konvergenzgarantien**: Theorem 5.1 beweist Konvergenz zu optimalen Lösungen
- ✅ **Kompressionsschranken**: Theorem 5.2 gibt fundamentale Grenzen für Kompression
- ✅ **Ternäre Gewichte**: Kompression auf {-1, 0, +1} → ~1/16 Originalgröße
- ✅ **PyTorch Integration**: Einfache API für bestehende Modelle
- ✅ **Mathematisch rigoros**: Alle Algorithmen mit Beweisen aus der Theorie

## 🚀 Installation

### Voraussetzungen

**Wichtig für Windows**: Visual Studio C++ Build Tools müssen installiert sein (siehe [CLAUDE.md](CLAUDE.md)).

```bash
# Python 3.8+
python --version

# Visual Studio C++ Build Tools (Windows)
# Download: https://visualstudio.microsoft.com/downloads/
```

### Installation aus Source

```bash
# Clone repository
git clone https://github.com/yourusername/IGQK.git
cd IGQK

# Install dependencies
pip install -r requirements.txt

# Install IGQK
pip install -e .
```

### Dependencies

- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `matplotlib >= 3.5.0`

## 📚 Schnellstart

### Beispiel: MNIST Training mit IGQK

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from igqk.integration.pytorch import IGQKTrainer


# 1. Define your neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)


# 3. Initialize IGQK Trainer
model = SimpleNet()
trainer = IGQKTrainer(
    model=model,
    train_loader=train_loader,
    hbar=0.1,          # Quantenunschärfe
    gamma=0.01,        # Dämpfung
    compression_type='ternary'  # {-1, 0, +1}
)


# 4. Train with quantum gradient flow
trainer.train(num_epochs=10)


# 5. Compress model to ternary weights
compressed_model = trainer.compress()


# 6. Evaluate
loss, accuracy = trainer.evaluate(test_loader)
print(f"Compressed Model Accuracy: {accuracy * 100:.2f}%")
```

### Vollständiges MNIST Beispiel

```bash
cd examples
python mnist_example.py
```

Dies führt vollständiges IGQK Training aus:
1. Training mit Quantengradientenfluss
2. Kompression zu ternären Gewichten
3. Evaluation und Vergleich
4. Visualisierung der Trainingsmetriken

## 🏗️ Architektur

```
igqk/
├── core/                          # Kernkomponenten
│   ├── manifold.py               # Statistische Mannigfaltigkeit + Fisher-Metrik
│   ├── quantum_state.py          # Dichtematrizen ρ
│   ├── evolution.py              # Quantengradientenfluss
│   └── measurement.py            # Messoperatoren
│
├── compression/                   # Kompressionsmethoden
│   ├── projection.py             # Optimale Projektion Π: M → N
│   ├── ternary.py                # Ternäre Kompression {-1, 0, +1}
│   ├── lowrank.py                # Low-Rank Kompression
│   └── sparse.py                 # Sparse Kompression
│
├── integration/                   # Framework-Integration
│   ├── pytorch.py                # PyTorch Optimizer & Trainer
│   └── optimizer.py              # Custom Optimizer Interface
│
├── geometry/                      # Geometrische Berechnungen
│   ├── fisher.py                 # Fisher-Informationsmetrik
│   ├── laplacian.py              # Laplace-Beltrami Operator
│   └── geodesic.py               # Geodäten auf Mannigfaltigkeiten
│
└── theory/                        # Theoretische Frameworks
    ├── hlwt.py                   # Hybrid Laplace-Wavelet Transform
    ├── tlgt.py                   # Ternary Lie Group Theory
    └── fchl.py                   # Fractional Calculus Hebbian Learning
```

## 🔬 Mathematische Theorie

### Theorem 5.1 (Konvergenz)

Der Quantengradientenfluss konvergiert zu einem stationären Zustand ρ* mit:

```
E_ρ*[L] ≤ min_{θ ∈ M} L(θ) + O(ℏ)
```

wobei ℏ die Quantenunschärfe ist.

### Theorem 5.2 (Kompressionsschranke)

Die minimale Verzerrung D bei Kompression auf k-dimensionale Untermannigfaltigkeit N:

```
D ≥ (n-k)/(2β) · log(1 + β·σ²_min)
```

Für ternäre Kompression (n → n/16):

```
D ≥ (15n/16)/(2β) · log(1 + β·σ²_min)
```

### Theorem 5.3 (Verschränkung & Generalisierung)

Verschränkte Quantenzustände über Layer verbessern Generalisierung:

```
E_gen ≤ E_train + O(√(I(A:B)/n))
```

wobei I(A:B) die Quanten-Mutual-Information ist.

**Vollständige mathematische Details**: Siehe [Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf](Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf)

## 🎛️ Hyperparameter

### Quantenparameter

- **ℏ (hbar)**: Quantenunschärfe (default: 0.1)
  - Höher → mehr Exploration
  - Niedriger → mehr Exploitation

- **γ (gamma)**: Dämpfungsparameter (default: 0.01)
  - Höher → schnellere Konvergenz
  - Niedriger → mehr Exploration

- **dt**: Zeitschritt für Integration (default: 0.01)

### Kompressionstypen

- **`ternary`**: Gewichte ∈ {-1, 0, +1} → ~1/16 Kompression
- **`lowrank`**: Low-Rank Matrix-Zerlegung
- **`sparse`**: Sparsity-basierte Kompression

## 📊 Prozess überwachen

### Quantum State Monitoring

```python
# Im Training Callback
def monitor(epoch, metrics):
    print(f"Epoch {epoch}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")  # Von Neumann Entropie
    print(f"  Purity: {metrics['purity']:.4f}")    # Tr(ρ²)
    print(f"  Trace: {metrics['trace']:.4f}")      # Sollte ~1.0 sein

trainer.train(num_epochs=10, callback=monitor)
```

### Wichtige Metriken

- **Entropy S(ρ) = -Tr(ρ log ρ)**: Maß für Quantenunsicherheit
  - S = 0: Reiner Zustand (keine Unsicherheit)
  - S > 0: Gemischter Zustand (Superposition)

- **Purity Tr(ρ²)**: Maß für Quantenreinheit
  - = 1: Reiner Zustand
  - < 1: Gemischter Zustand

- **Trace Tr(ρ)**: Sollte immer 1.0 sein (Normierung)

## 🐛 Fehlerbehandlung

### Häufige Fehler

#### 1. "Always getting errors" (aus CLAUDE.md)

**Lösung**: Überprüfen Sie, dass Visual Studio C++ Build Tools installiert sind:

```bash
# Windows
# Download und installieren: https://visualstudio.microsoft.com/downloads/
```

#### 2. Negative Eigenvalues

```python
# Automatisch behandelt in QuantumState:
# - Negative Werte werden auf 0 geclampt
# - Automatische Renormierung
```

#### 3. Fisher Matrix singular

```python
# Regularisierung ist eingebaut:
# fisher_inv = torch.linalg.inv(fisher + 1e-4 * I)
```

## 📈 Performance

### Komplexität (aus Algorithmus 1)

- **Quantenupdate**: O(n² · d²) pro Schritt
  - n = Parameteranzahl
  - d = Rang der Dichtematrix

- **Mit Low-Rank Approximation** (d = O(log n)):
  - O(T · n² · log² n) gesamt

### Skalierung auf große Modelle

Für Modelle mit 1B+ Parametern:
- Layer-weise IGQK Anwendung
- Block-diagonale Fisher-Metrik Approximation
- Parallele Quantenupdates über Layer

## 🔗 Theoretische Verbindungen

IGQK vereint drei ursprüngliche Frameworks:

### 1. HLWT als Spezialfall (Proposition 6.1)
HLWT ist Fourier-Transformation des Quantengradientenflusses in lokalen Koordinaten.

### 2. TLGT als Spezialfall (Proposition 6.2)
Ternäre Lie-Gruppe G₃ ist diskrete Untergruppe der Quantensymmetriegruppe auf M.

### 3. FCHL als Spezialfall (Proposition 6.3)
Fraktionales Hebbian Learning verwendet fraktionalen Laplace-Beltrami Operator H = -(-Δ_M)^α.

## 🔍 Offene Forschungsfragen

1. **Optimales ℏ**: Wie wählt man Quantenunschärfe optimal?
2. **Verschränkungsstruktur**: Welche Verschränkungsmuster sind optimal?
3. **Quantenvorteil**: Gibt es echten Speedup gegenüber klassischen Methoden?
4. **Hardware**: Kann IGQK auf Quantencomputern laufen?
5. **Universalität**: Gilt IGQK für alle Architekturen (CNNs, Transformers, etc.)?

## 📝 Zitierung

Wenn Sie IGQK in Ihrer Forschung verwenden, bitte zitieren Sie:

```bibtex
@misc{igqk2024,
  title={IGQK: Information-Geometric Quantum Compression for Neural Networks},
  author={IGQK Research Team},
  year={2024},
  note={Theoretisches Framework basierend auf Informationsgeometrie und Quantenmechanik}
}
```

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

## 🤝 Beitragen

Contributions sind willkommen! Bitte siehe [CONTRIBUTING.md](CONTRIBUTING.md).

## 📚 Weiterführende Literatur

- **Mathematische Details**: `Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf`
- **Meta-Analyse**: `KI-Analyse_Welche_mathematischen_Wege_machen_Sinn.pdf`
- **Projekt-Richtlinien**: [CLAUDE.md](CLAUDE.md)

## 🆘 Support

- **Issues**: GitHub Issues für Bugs und Feature-Requests
- **Diskussionen**: GitHub Discussions für Fragen
- **Email**: support@igqk-project.org

---

**IGQK** - Wo Informationsgeometrie auf Quantenmechanik trifft 🎯🔬
