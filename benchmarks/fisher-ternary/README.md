# Empirischer Probe-Benchmark: Information-geometrische Selektion bei Ternär-Quantisierung

**Frage:** Bringt die information­geometrische *Auswahl* (Fisher-Diagonale entscheidet, welche
Gewichte vor der Ternarisierung in fp32 geschützt werden) bei **gleichem Kompressionsbudget**
messbar bessere Perplexity als uniforme Magnitude-Ternarisierung (BitNet-b1.58-absmean)?

Dies ist der „eine entscheidende nächste Schritt" aus `MARKTRELEVANZ.md`, hier minimal umgesetzt
und **real ausgeführt** (nicht nur behauptet).

## Setup (ehrlich, klein, CPU — richtungsweisend, nicht publikationsreif)

- **Modell:** `gpt2` (124M), echtes Transformer-LM.
- **Held-out:** WikiText-2-raw Test, Perplexity über 60 Fenster à 512 Token (nicht-überlappend).
- **Quantisiert:** alle 48 2D-Conv1D/Linear-Gewichte der Transformer-Blöcke (84.9M Params).
- **absmean-Ternär:** `γ = mean(|W|); Q = γ·clip(round(W/γ), −1, +1)` (BitNet-b1.58-Regel).
- **Schutz-Budget p:** pro Tensor die wichtigsten p der Gewichte in fp32 behalten, Rest ternär.
  Drei Salienz-Scores bei gleichem p verglichen:
  - `magnitude` = |Wᵢ|
  - `quant_err` = (Wᵢ − Qᵢ)²
  - `fisher`    = Fᵢᵢ · (Wᵢ − Qᵢ)²   ← die info-geometrische / OBD-artige Größe
- **Fisher-Diagonale** Fᵢᵢ = Mittel über 24 Kalibrier­sequenzen von (∂L/∂wᵢ)² (empirische Fisher).
- **Effektive Bits** = p·32 + (1−p)·1.58.

Skript: [`fisher_ternary_bench.py`](./fisher_ternary_bench.py) — deterministisch (seed 0), CPU,
reproduzierbar mit `python3 fisher_ternary_bench.py`.

## Ergebnis

| Bedingung | Schutz | bit/w | PPL | vs. uniform |
|---|---:|---:|---:|---:|
| FP32 (Referenz) | – | 32.0 | **37.93** | – |
| uniform absmean-Ternär | 0 % | 1.58 | 1 044 763 | – |
| Schutz nach magnitude | 1 % | 1.88 | 4 286.7 | −99.6 % |
| Schutz nach quant_err | 1 % | 1.88 | 4 286.7 | −99.6 % |
| **Schutz nach fisher** | 1 % | 1.88 | **2 429.0** | **−99.8 %** |
| Schutz nach magnitude | 5 % | 3.10 | 2 350.2 | −99.8 % |
| Schutz nach quant_err | 5 % | 3.10 | 2 350.2 | −99.8 % |
| **Schutz nach fisher** | 5 % | 3.10 | **1 323.9** | **−99.9 %** |

Bei **gleichem Budget** liefert die Fisher-Selektion **~43–44 % niedrigere Perplexity** als die
Magnitude-Selektion (1 %: 2 429 vs. 4 287; 5 %: 1 324 vs. 2 350).

## Interpretation (schonungslos)

**Positiv / neu:** Die information­geometrische Auswahl trägt **reales, konsistentes Signal** — genau
der Mechanismus, den die IGQK-Theorie behauptet, aber den der Repo-Code nie implementiert hat (dort
wird Fisher berechnet, aber nicht zur Quantisierungs­entscheidung genutzt). Erst die Krümmung Fᵢᵢ
bricht die Gleichheit zwischen `magnitude` und `quant_err` auf.

**Caveats, ohne die es Marketing wäre:**
1. **Alle Ternär-Varianten sind unbrauchbar** vs. FP32 (1 324 vs. 38 PPL ≈ 35× schlechter). Naive
   Post-Training-Ternarisierung zerstört GPT-2 — bestätigt, dass BitNet QAT/From-Scratch-Training
   braucht. Fisher macht es „weniger katastrophal", nicht „gut".
2. **magnitude == quant_err** ist ein erwartetes Artefakt: im extremen Tail (top 1–5 %) sind |W| und
   (W−Q)² monoton äquivalent.
3. **Kein GPTQ/AWQ-Baseline** hier — Aussage gilt nur *innerhalb* naiver Ternär-PTQ, **nicht** „schlägt GPTQ".
4. Kleines Modell, CPU, nicht-überlappende Fenster, begrenzte Kalibrierung.

## Nächste, wirklich entscheidende Schritte

1. **GPTQ/AWQ-Baseline** hinzufügen — schließt error-compensation den Fisher-Vorsprung?
2. **QAT / Kurz-Finetuning mit STE** nach der Quantisierung — bringt es Ternär in brauchbare Nähe von FP32?
3. **Skalierung** auf Qwen2.5-0.5B + per-channel-Scaling; Perplexity *und* gemessenen RAM-Footprint berichten.

---
*Erstellt durch automatisiertes Experiment. Zahlen reproduzierbar via beigefügtem Skript (seed 0, CPU).*
