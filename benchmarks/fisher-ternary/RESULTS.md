# GPTQ-Baseline + Joule/Token — Ergebnisse (entscheidender Test)

Antwort auf: *„Schlägt die information­geometrische Fisher-Selektion echtes GPTQ — und
gibt es einen Tokens/Watt-Hebel?"* Alles real ausgeführt (GPT-2 124M, WikiText-2,
i5-13500 CPU, RAPL-Energie via sudo). Skripte: `partA_gptq_vs_fisher.py`,
`partB2_energy_controlled.py` (seed 0, reproduzierbar).

## Teil A — Qualität vs. Bits (Perplexity, niedriger = besser)

| Methode | ~bit/w | PPL | vs FP32 |
|---|---:|---:|---:|
| FP32 (Referenz) | 32.0 | **37.93** | 1.00× |
| absmean-Ternär (uniform) | 1.58 | 1 044 763 | 27 544× |
| GPTQ-2bit (= Ternär + Fehlerkomp.) | 2.00 | 72 556 | 1 913× |
| GPTQ-3bit | 3.00 | **159.87** | 4.21× |
| **Fisher-Ternär 5 %** (meine Methode) | 3.10 | 1 323.86 | 34.9× |
| RTN-int4 (ohne Fehlerkomp.) | 4.00 | 1 845.76 | 48.7× |
| **GPTQ-4bit** | 4.00 | **42.63** | **1.12×** |

**Befunde (schonungslos):**
1. **GPTQ dominiert die Frontier vollständig.** GPTQ-4bit ist praktisch FP32-gleich (1.12×). Bei *gleichen ~3 Bit* ist GPTQ-3bit (159.9) **~8× besser** als meine Fisher-Ternär (1324).
2. **Fehlerkompensation ≫ Metrik-Selektion.** RTN-int4 (1846) vs. GPTQ-int4 (42.6) bei *identischen 4 Bit*: der Hessian-basierte Fehlerausgleich bringt **43×** bessere PPL. *Dort* liegt der Wert — nicht in der Auswahl, welche Gewichte man schützt.
3. **Ternär-PTQ ist tot**, egal mit welcher Methode (selbst GPTQ-2bit kollabiert auf 72 556). Ternär braucht QAT/From-Scratch (BitNet), kein Post-Training.
4. Der frühere „Fisher schlägt Magnitude um 40 %"-Effekt ist **real, aber irrelevant**: er verbessert eine Methode (Ternär-PTQ), die ohnehin nicht auf der Frontier liegt.

→ **Es gibt keinen Quantisierungs-Moat.** Die beste Methode (GPTQ/AWQ-int4) ist frei verfügbar und schlägt alles Eigene um Größenordnungen.

## Teil B — Echte Joule/Token (RAPL, i5-13500)

| Größe | Wert |
|---|---|
| Idle-Package-Leistung | 20.0 W |
| fp32-Decode | **1.49 ± 0.04 J/Token** · 39–45 tok/s · 65 W · **€0.124 / 1 Mio Token** @ €0.30/kWh |
| int4-Werte-als-f32 (Repo-Stil) | 1.30 J/Token — Differenz **innerhalb des Rauschens**, kein echter Gewinn |

**Befunde:**
1. **f32-gespeichertes „Quant" spart ~0 Energie** (gemessen). Beweis a priori: quantisierte Werte als f32 → byte-identisches dichtes Matmul. Die gemessenen ±10 % sind Turbo/Thermik-Artefakt auf einem geteilten Desktop — **Lektion: kleine Tokens/Watt-Deltas ohne isoliertes Mess-Rig sind wertlos.**
2. **Bandbreiten-Modell** (Gewichtsbewegung/Token, ~20 pJ/Byte DRAM):

   | Speicherung | mJ/Token | Faktor |
   |---|---:|---:|
   | fp16 (248 MB/Tok) | 4.96 | 1× |
   | int4 *gepackt* | 1.24 | 4× |
   | 1.58-bit *gepackt* | 0.49 | **10×** |

3. **Der einzige Tokens/Watt-Hebel** ist *gepackter* Sub-Byte-Speicher **+ ein Kernel, der gepackt liest**. Genau das liefern bitnet.cpp / GPTQ-Kernel bereits — und genau das fehlt in diesen Repos (Ternär als f32, Packing nur beim Datei-Export, „add-Kernel" toter Test-Code).

## Gesamturteil

- **Quantisierung als Produkt: tot.** Kein Moat; Commodity-GPTQ schlägt alles Eigene.
- **Fisher-Selektion:** reales, aber kleines Forschungs-Nugget, von GPTQ geschlagen → höchstens eine ehrliche Notiz, kein Geschäft.
- **Tokens/Watt-Hebel:** existiert nur im gepackten Kernel — den bitnet.cpp schon hat.
- **Wo Wert bleibt:** nicht im Produkt, sondern im *Beweis der Fähigkeit* — dieser Test selbst (eigenes GPTQ, echte RAPL-Energie, ehrliche Frontier) ist das verkäufliche Artefakt für Talent/Consulting. Plus die A2A-Agent-Orchestrierung als getrennte, nicht-Tokens/Watt-Wette.

---
*Reproduzierbar via beigefügten Skripten. Energie: Intel RAPL package-0. Modell GPT-2, WikiText-2-raw.*
