# IGQK Framework - Mathematische Grundlagen

## Übersicht

Dieses Dokument erklärt die mathematischen Grundlagen von IGQK Schritt für Schritt, von den Basics bis zu den fortgeschrittenen Theoremen.

## 1. Statistische Mannigfaltigkeiten

### 1.1 Definition

Eine **statistische Mannigfaltigkeit** ist ein Raum von Wahrscheinlichkeitsverteilungen, ausgestattet mit einer Riemannschen Metrik.

**Formal**: Sei M = {p(y|x, θ) : θ ∈ Θ} eine Familie von Wahrscheinlichkeitsverteilungen, parametrisiert durch θ ∈ Θ ⊂ ℝⁿ.

### 1.2 Fisher-Informationsmatrix

Die **Fisher-Metrik** auf M ist definiert als:

```
g_ij(θ) = E_{y~p(·|x,θ)} [∂_i log p(y|x,θ) · ∂_j log p(y|x,θ)]
```

wobei ∂_i = ∂/∂θ_i.

**Eigenschaften**:
1. **Symmetrisch**: g_ij = g_ji
2. **Positiv-semidefinit**: vᵀGv ≥ 0 für alle v
3. **Invariant**: Unabhängig von Parametrisierung

**Intuition**: g_ij misst, wie stark sich die Verteilung p ändert, wenn man θ_i und θ_j verändert.

### 1.3 Natürlicher Gradient

Der **natürliche Gradient** ist definiert als:

```
∇̃L = G⁻¹∇L
```

wobei G die Fisher-Matrix und ∇L der normale Gradient ist.

**Warum ist das besser?**

Der natürliche Gradient ist die Richtung des steilsten Abstiegs bezüglich der **KL-Divergenz**, nicht der Euklidischen Distanz:

```
θ_{t+1} = argmin_θ [L(θ) + (1/2η)||θ - θ_t||²_G]
```

wobei ||·||_G die Norm bezüglich der Fisher-Metrik ist.

### 1.4 Geodätische Distanz

Die **geodätische Distanz** zwischen zwei Punkten θ₁, θ₂ auf M ist:

```
d_M(θ₁, θ₂) = min_{γ} ∫₀¹ √(γ̇ᵀG(γ(t))γ̇) dt
```

wobei γ:[0,1]→M ein Pfad mit γ(0)=θ₁ und γ(1)=θ₂ ist.

**Intuition**: Die kürzeste Kurve auf der gekrümmten Mannigfaltigkeit.

## 2. Quantenzustände

### 2.1 Dichtematrizen

Ein **Quantenzustand** ist eine Dichtematrix ρ ∈ ℂⁿˣⁿ mit:

1. **Hermitisch**: ρ = ρ†
2. **Positiv-semidefinit**: ρ ≥ 0
3. **Normalisiert**: Tr(ρ) = 1

**Spektralzerlegung**:
```
ρ = Σᵢ λᵢ |ψᵢ⟩⟨ψᵢ|
```

wobei λᵢ ≥ 0 Eigenwerte und |ψᵢ⟩ Eigenvektoren sind.

### 2.2 Erwartungswerte

Der **Erwartungswert** eines Observablen O ist:

```
⟨O⟩_ρ = Tr(ρO)
```

**Für neuronale Netze**: O = θ (Parameter), also:
```
⟨θ⟩_ρ = Tr(ρθ) = Σᵢ λᵢ⟨ψᵢ|θ|ψᵢ⟩
```

### 2.3 Von-Neumann-Entropie

Die **Entropie** eines Quantenzustands ist:

```
S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log λᵢ
```

**Eigenschaften**:
- S(ρ) = 0 ⟺ ρ ist reiner Zustand (|ψ⟩⟨ψ|)
- S(ρ) = log n ⟺ ρ ist maximal gemischt (I/n)

**Intuition**: Misst die "Unschärfe" oder "Mischung" des Zustands.

### 2.4 Reinheit

Die **Reinheit** ist:

```
P(ρ) = Tr(ρ²) = Σᵢ λᵢ²
```

**Eigenschaften**:
- P(ρ) = 1 ⟺ reiner Zustand
- P(ρ) = 1/n ⟺ maximal gemischter Zustand

**Beziehung zur Entropie**: Hohe Reinheit ⟺ niedrige Entropie.

### 2.5 Quantenverschränkung

Für ein zusammengesetztes System ρ_AB ist die **Verschränkung** gemessen durch:

```
I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
```

wobei ρ_A = Tr_B(ρ_AB) die partielle Spur ist.

**Intuition**: Wie viel Information teilen A und B?

## 3. Quantengradientenfluss (QGF)

### 3.1 Lindblad-Gleichung

Die Evolution des Quantenzustands folgt der **Lindblad-Gleichung**:

```
dρ/dt = -i[H, ρ] + Σₖ (LₖρL†ₖ - (1/2){L†ₖLₖ, ρ})
```

**Für IGQK** vereinfachen wir zu:

```
dρ/dt = -i[H, ρ] - γ{∇L, ρ}
```

### 3.2 Komponenten

**Unitärer Teil**: `-i[H, ρ]`
- H = Hamiltonian (Energieoperator)
- [H, ρ] = Hρ - ρH (Kommutator)
- Erhält Reinheit: d/dt Tr(ρ²) = 0

**Dissipativer Teil**: `-γ{∇L, ρ}`
- ∇L = Gradient der Loss-Funktion
- {A, B} = AB + BA (Antikommutator)
- Reduziert Energie: d/dt Tr(ρH) < 0

### 3.3 Diskretisierung

Für numerische Integration verwenden wir:

```
ρ(t+dt) = ρ(t) - dt·i[H, ρ(t)] - dt·γ{∇L, ρ(t)}
```

mit Normalisierung:
```
ρ(t+dt) ← ρ(t+dt) / Tr(ρ(t+dt))
```

### 3.4 Hamiltonian-Wahl

Der Hamiltonian H wird gewählt als:

```
H = ∇L·∇Lᵀ / ||∇L||²
```

**Intuition**: Oszillation in Richtung des Gradienten (Exploration).

## 4. Geometrische Projektion

### 4.1 Optimale Projektion

Gegeben eine Mannigfaltigkeit M und eine Untermannigfaltigkeit N ⊂ M, ist die **optimale Projektion**:

```
Π_N(θ) = argmin_{w ∈ N} d_M(θ, w)
```

wobei d_M die geodätische Distanz auf M ist.

### 4.2 Ternäre Projektion

Für ternäre Gewichte w ∈ {-α, 0, +α}ⁿ:

**Optimales α** (minimiert L2-Distortion):
```
α* = E[|θ_i|] / P(θ_i ≠ 0)
```

**Projektion**:
```
w_i = { -α  wenn θ_i < -α/2
      {  0  wenn |θ_i| ≤ α/2
      { +α  wenn θ_i > α/2
```

### 4.3 Binäre Projektion

Für binäre Gewichte w ∈ {-α, +α}ⁿ:

**Optimales α**:
```
α* = E[|θ_i|]
```

**Projektion**:
```
w_i = α · sign(θ_i)
```

### 4.4 Sparse Projektion

Für sparse Gewichte (Sparsität s):

**Schwellwert**:
```
τ = k-th largest |θ_i|, wobei k = ⌊(1-s)·n⌋
```

**Projektion**:
```
w_i = { θ_i  wenn |θ_i| ≥ τ
      { 0    sonst
```

## 5. Haupttheoreme

### Theorem 5.1: Konvergenz des QGF

**Aussage**: Unter milden Bedingungen konvergiert der Quantengradientenfluss zu einem ε-optimalen Zustand:

```
E_ρ*[L] ≤ min_θ L(θ) + O(ℏ)
```

**Beweisskizze**:
1. Zeige dass Tr(ρH) monoton fällt (Lyapunov-Funktion)
2. Nutze Dissipation: d/dt Tr(ρH) = -γ||∇L||²_ρ ≤ 0
3. Konvergenz zu stationärem Punkt: ∇L = 0 (bis auf O(ℏ))

### Theorem 5.2: Rate-Distortion-Schranke

**Aussage**: Die minimale Distortion D bei Kompression von n auf k Parameter ist:

```
D ≥ C · (n-k)² / n
```

für eine Konstante C > 0.

**Beweisskizze**:
1. Nutze Informationstheorie: D ≥ 2^(-R) wobei R = log(n/k)
2. Geometrische Interpretation: Volumen der Kompressionsmannigfaltigkeit
3. Kombiniere zu quadratischer Schranke

### Theorem 5.3: Verschränkung und Generalisierung

**Aussage**: Für ein verschränktes System ρ_AB ist der Generalisierungsfehler:

```
E_gen ≤ O(√((C_A + C_B - I(A:B)) / m))
```

wobei C_A, C_B Komplexitäten und I(A:B) Verschränkung ist.

**Intuition**: Verschränkung reduziert effektive Komplexität!

**Beweisskizze**:
1. Nutze PAC-Bayes-Schranke
2. Effektive Dimension = C_A + C_B - I(A:B)
3. Verschränkung "teilt" Information zwischen A und B

## 6. Praktische Approximationen

### 6.1 Low-Rank-Approximation

Für große Systeme approximieren wir ρ durch:

```
ρ ≈ Σᵢ₌₁ʳ λᵢ |ψᵢ⟩⟨ψᵢ|
```

mit r ≪ n.

**Komplexität**: O(nr²) statt O(n²)

### 6.2 Diagonale Fisher-Approximation

Approximiere Fisher-Matrix durch Diagonale:

```
G ≈ diag(g₁₁, g₂₂, ..., g_nn)
```

**Komplexität**: O(n) statt O(n²)

### 6.3 Ensemble-Approximation

Approximiere Erwartungswerte durch Monte-Carlo:

```
E_ρ[f] ≈ (1/K) Σₖ₌₁ᴷ f(θₖ), θₖ ~ ρ
```

**Vorteil**: Parallelisierbar, keine explizite Dichtematrix nötig.

## 7. Mathematische Zusammenhänge

### 7.1 IGQK vereinheitlicht bekannte Methoden

| Methode | IGQK-Parameter |
|---------|----------------|
| SGD | ℏ=0, γ=0, G=I |
| Natural Gradient | ℏ=0, γ=0 |
| Simulated Annealing | ℏ groß → klein |
| Ensemble Methods | ℏ > 0, mehrere Samples |

### 7.2 Verbindung zur Physik

IGQK ist inspiriert von:
- **Quantenmechanik**: Schrödinger-Gleichung
- **Statistische Mechanik**: Boltzmann-Verteilung
- **Thermodynamik**: Freie Energie als Lyapunov-Funktion

### 7.3 Verbindung zur Informationstheorie

- **Fisher-Information** ≈ Krümmung der KL-Divergenz
- **Von-Neumann-Entropie** ≈ Shannon-Entropie für Quantensysteme
- **Rate-Distortion** ≈ Kompression-Genauigkeit-Tradeoff

## Zusammenfassung

Die Mathematik von IGQK basiert auf drei Pfeilern:

1. **Informationsgeometrie**: Fisher-Metrik, natürlicher Gradient, geodätische Distanz
2. **Quantenmechanik**: Dichtematrizen, Lindblad-Gleichung, Verschränkung
3. **Optimierung**: Konvergenzbeweise, Rate-Distortion, Projektionen

Diese Konzepte werden **vereinheitlicht** durch die Behandlung neuronaler Netze als Quantensysteme auf statistischen Mannigfaltigkeiten.

**Das Ergebnis**: Ein theoretisch fundiertes Framework mit praktischer Leistung!
