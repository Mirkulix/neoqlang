# ⚡ IGQK v4.0 - QUICK START GUIDE

**Sofort loslegen mit v4.0 Development!**

---

## 🎯 DEIN ERSTER SCHRITT: MODEL-KLASSEN

### Problem verstehen

```python
# Das funktioniert NICHT (quantum_llm_trainer.py:104):
def _create_model(self):
    if self.config.model_type == 'GPT':
        from ...models.gpt import QuantumGPT  # ❌ FEHLT!
        return QuantumGPT(self.config)
```

**Warum?** Der Ordner `igqk_v4/models/` existiert nicht!

---

## 📝 TO-DO LISTE (Diese Woche)

### Schritt 1: Ordner erstellen (2 Minuten)

```bash
cd C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_v4

# Erstelle models/ Ordner
mkdir models
cd models

# Erstelle __init__.py
echo. > __init__.py
```

---

### Schritt 2: QuantumGPT implementieren (4 Stunden)

**Erstelle: `models/gpt.py`**

Kopiere den Code aus `IGQK_V4_MASTER_PLAN.md` → Abschnitt "1.1 Erstelle: igqk_v4/models/gpt.py"

Oder hier nochmal kurz:

```python
"""Quantum-Enhanced GPT Model."""

import torch
import torch.nn as nn
from ..quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from ..theory.tlgt.ternary_lie_group import TernaryLinear
import math


class QuantumMultiHeadAttention(nn.Module):
    """Multi-Head Attention mit ternären Weights."""

    def __init__(self, d_model, n_heads, dropout=0.1, use_ternary=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        # Q, K, V projections (ternär wenn train_compressed=True)
        LinearLayer = TernaryLinear if use_ternary else nn.Linear

        self.q_proj = LinearLayer(d_model, d_model)
        self.k_proj = LinearLayer(d_model, d_model)
        self.v_proj = LinearLayer(d_model, d_model)
        self.out_proj = LinearLayer(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Q, K, V
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, attn


class QuantumFeedForward(nn.Module):
    """FFN mit ternären Weights."""

    def __init__(self, d_model, d_ff, dropout=0.1, use_ternary=True):
        super().__init__()
        LinearLayer = TernaryLinear if use_ternary else nn.Linear

        self.fc1 = LinearLayer(d_model, d_ff)
        self.fc2 = LinearLayer(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class QuantumTransformerBlock(nn.Module):
    """Ein Transformer Block."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_ternary=True):
        super().__init__()
        self.attn = QuantumMultiHeadAttention(d_model, n_heads, dropout, use_ternary)
        self.ffn = QuantumFeedForward(d_model, d_ff, dropout, use_ternary)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN style
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, attn_weights


class QuantumGPT(nn.Module):
    """Quantum-Enhanced GPT."""

    def __init__(self, config: QuantumTrainingConfig):
        super().__init__()

        self.config = config
        use_ternary = config.train_compressed

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                use_ternary
            )
            for _ in range(config.n_layers)
        ])

        # Final
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embed.weight

        # Init
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"✅ QuantumGPT: {config.n_layers} layers, {n_params:,} params")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape

        # Embeddings
        tok_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb

        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device))
        causal_mask = causal_mask.view(1, 1, T, T)

        # Transformer
        for block in self.blocks:
            x, _ = block(x, mask=causal_mask)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """Auto-regressive generation."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# Export
__all__ = ['QuantumGPT']
```

**Speichern als:** `igqk_v4/models/gpt.py`

---

### Schritt 3: __init__.py erstellen

**Erstelle: `models/__init__.py`**

```python
"""IGQK v4.0 Model Library."""

from .gpt import QuantumGPT

__all__ = ['QuantumGPT']
```

---

### Schritt 4: Testen! (5 Minuten)

**Erstelle: `test_quantum_gpt.py` (im Hauptordner)**

```python
"""Test QuantumGPT Model."""

import torch
import sys
sys.path.append('igqk_v4')

from quantum_training.trainers.quantum_training_config import QuantumTrainingConfig
from models.gpt import QuantumGPT

print("🧪 Testing QuantumGPT\n")

# Config
config = QuantumTrainingConfig(
    model_type='GPT',
    n_layers=6,
    n_heads=8,
    d_model=512,
    d_ff=2048,
    vocab_size=10000,
    max_seq_len=128,
    train_compressed=True,  # Ternary!
)

# Model
print("Creating model...")
model = QuantumGPT(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Test forward
print("Testing forward pass...")
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

logits = model(input_ids)
print(f"  Input: {input_ids.shape}")
print(f"  Output: {logits.shape}")
assert logits.shape == (batch_size, seq_len, config.vocab_size)
print("  ✅ Forward pass OK!\n")

# Test generation
print("Testing generation...")
start = torch.randint(0, config.vocab_size, (1, 5))
generated = model.generate(start, max_new_tokens=10)
print(f"  Start: {start.shape}")
print(f"  Generated: {generated.shape}")
assert generated.shape == (1, 15)
print("  ✅ Generation OK!\n")

print("✅ ALL TESTS PASSED!")
```

**Run:**

```bash
cd C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package
python test_quantum_gpt.py
```

**Erwartete Ausgabe:**

```
🧪 Testing QuantumGPT

Creating model...
✅ QuantumGPT: 6 layers, 37,593,600 params
Parameters: 37,593,600

Testing forward pass...
  Input: torch.Size([2, 10])
  Output: torch.Size([2, 10, 10000])
  ✅ Forward pass OK!

Testing generation...
  Start: torch.Size([1, 5])
  Generated: torch.Size([1, 15])
  ✅ Generation OK!

✅ ALL TESTS PASSED!
```

---

## 🚀 NÄCHSTE SCHRITTE

### Wenn QuantumGPT funktioniert:

**Option A: BERT implementieren** (ähnlich wie GPT, aber bidirectional)

```python
# models/bert.py
class QuantumBERT(nn.Module):
    # Wie GPT, aber ohne causal mask!
    pass
```

**Option B: ViT implementieren** (für Bilder)

```python
# multimodal/vision/vision_encoder.py
class QuantumVisionEncoder(nn.Module):
    # ViT mit Patches
    pass
```

**Option C: Training testen**

```python
# test_training.py
from quantum_training.trainers.quantum_llm_trainer import QuantumLLMTrainer

config = QuantumTrainingConfig(
    model_type='GPT',
    n_layers=6,
    n_heads=8,
    d_model=512,
    use_quantum=True,
    train_compressed=True,
    use_hlwt=True,
    use_tlgt=True,
)

trainer = QuantumLLMTrainer(config)

# Train!
# model = trainer.fit(train_data, val_data, n_epochs=10)
```

---

## 📊 PROGRESS TRACKER

```
□ Models erstellt (models/)
  □ __init__.py
  □ gpt.py          ← STARTE HIER!
  □ bert.py
  □ vit.py

□ Test läuft
  □ test_quantum_gpt.py funktioniert
  □ Forward pass OK
  □ Generation OK

□ Training funktioniert
  □ Trainer kann Model erstellen
  □ Training Loop läuft
  □ HLWT/TLGT integriert

□ Multi-Modal
  □ Vision Encoder
  □ Language Encoder
  □ Fusion
```

---

## ⚡ SPEED-RUN (4 Stunden)

```
09:00 - 10:00  │  Ordner erstellen + gpt.py kopieren
10:00 - 11:00  │  test_quantum_gpt.py schreiben + debuggen
11:00 - 12:00  │  Trainer-Integration testen
12:00 - 13:00  │  MNIST Demo implementieren
```

**Ziel:** QuantumGPT funktioniert zu 100%!

---

## 🎯 WAS DU JETZT TUN SOLLTEST

### Sofort (10 Minuten):

1. Öffne VS Code / PyCharm
2. Navigiere zu `C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_v4`
3. Erstelle `models/` Ordner
4. Kopiere `gpt.py` Code (siehe oben)
5. Run `python test_quantum_gpt.py`

### Heute (4 Stunden):

1. QuantumGPT fertigstellen
2. Tests schreiben
3. Integration mit Trainer
4. Dokumentieren

### Diese Woche (2 Tage):

1. BERT + ViT implementieren
2. Training testen
3. HLWT/TLGT/FCHL Integration
4. Alpha-Version bereit

---

## 🔧 TROUBLESHOOTING

### Import Error?

```python
# Wenn ImportError: No module named 'quantum_training'
import sys
sys.path.append('igqk_v4')
```

### TernaryLinear nicht gefunden?

```python
# Prüfe ob theory/ Ordner existiert:
ls igqk_v4/theory/tlgt/ternary_lie_group.py

# Sollte da sein! ✅
```

### Visual Studio C++ Error?

```bash
# Installiere VS Build Tools oder:
conda install -c conda-forge scipy pywavelets
```

---

## 📚 HILFREICHE DOKUMENTE

```
📄 IGQK_V4_MASTER_PLAN.md
   → Vollständiger Code für alle Features

📄 IGQK_VOLLSTAENDIGE_ANALYSE_V4.md
   → Detaillierte Analyse

📄 SCHNELL_ÜBERSICHT_V4.md
   → Quick Overview

📄 IMPLEMENTIERUNGS_CHECKLISTE_V4.md
   → Alle TODOs
```

---

## 🎉 SUCCESS METRICS

**Du hast es geschafft wenn:**

- ✅ `test_quantum_gpt.py` läuft ohne Fehler
- ✅ Forward pass funktioniert
- ✅ Generation produziert Output
- ✅ Ternary weights funktionieren
- ✅ Trainer kann Model erstellen

**Dann bist du bereit für Phase 2: Multi-Modal!**

---

**🚀 JETZT STARTEN!**

```bash
cd C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_v4
mkdir models
# ... und los geht's!
```

---

**Letzte Aktualisierung:** 2026-02-05

**Nächster Schritt:** Implementiere `models/gpt.py`
