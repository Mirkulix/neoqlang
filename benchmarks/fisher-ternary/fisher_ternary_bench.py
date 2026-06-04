#!/usr/bin/env python3
"""
Decisive mini-experiment for the IGQK / qlang claim:

  Does an INFORMATION-GEOMETRIC SELECTION (Fisher-diagonal decides which weights
  to protect from ternarization) beat uniform magnitude-threshold ternarization
  (BitNet b1.58 absmean) at the SAME compression budget?

Setup (honest, small, CPU, post-training quantization -- directional probe, not a paper):
  - Model: gpt2 (124M), a real transformer LM.
  - Held-out test: WikiText-2 perplexity (sliding window).
  - Quantize all 2D Conv1D/Linear weights in the transformer blocks.
  - absmean ternary:  gamma = mean(|W|);  Q = gamma * clip(round(W/gamma), -1, 1).
  - "Protection" budget p: keep top-p of weights (per tensor) in fp32, ternarize the rest.
    Three salience scores compared at equal p:
       magnitude : |W_i|
       quant_err : (W_i - Q_i)^2
       fisher    : F_ii * (W_i - Q_i)^2     <-- the info-geometric / OBD-style score
    F_ii = empirical Fisher diagonal = mean over calibration data of (dL/dw_i)^2.
  - Effective bits = p*32 + (1-p)*1.58 ; we report PPL vs bits so comparisons are fair.
"""
import math, time, sys
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(min(16, (torch.get_num_threads() or 8)))
DEV = "cpu"
MODEL = "gpt2"
SEQ = 512
N_WINDOWS = 60          # ~30k tokens of WikiText-2 test
CALIB_SEQS = 24         # calibration sequences for the Fisher diagonal
PROTECT = [0.0, 0.01, 0.05]  # protected fraction (fp32) per tensor

print(f"[load] {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV)
model.train(False)

# --- identify quantizable weights: 2D weights of Conv1D / Linear in the blocks ---
def is_target(name, mod):
    if not hasattr(mod, "weight") or mod.weight is None: return False
    w = mod.weight
    if w.dim() != 2: return False
    if type(mod).__name__ not in ("Conv1D", "Linear"): return False
    if "wte" in name or "wpe" in name or "lm_head" in name: return False
    return True

targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
print(f"[targets] {len(targets)} weight tensors, "
      f"{sum(m.weight.numel() for _,m in targets)/1e6:.1f}M quantized params", flush=True)
orig = {n: m.weight.detach().clone() for n, m in targets}

# --- data ---
print("[data] WikiText-2", flush=True)
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
test_ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0]
calib_text = "\n\n".join(t for t in train["text"] if len(t) > 200)[:200000]
calib_ids = tok(calib_text, return_tensors="pt").input_ids[0]

@torch.no_grad()
def perplexity(m):
    m.train(False); nll, ntok = 0.0, 0
    for w in range(N_WINDOWS):
        s = w * SEQ
        ids = test_ids[s:s+SEQ]
        if ids.numel() < 2: break
        ids = ids.unsqueeze(0)
        out = m(ids, labels=ids)
        n = ids.numel() - 1
        nll += out.loss.item() * n; ntok += n
    return math.exp(nll / ntok)

def absmean_ternary(W):
    gamma = W.abs().mean().clamp_min(1e-8)
    return torch.clamp(torch.round(W / gamma), -1, 1) * gamma

# --- Fisher diagonal via accumulated squared gradients on calibration data ---
def fisher_diagonal():
    fisher = {n: torch.zeros_like(orig[n]) for n, _ in targets}
    nseq = 0
    for k in range(CALIB_SEQS):
        s = k * SEQ
        ids = calib_ids[s:s+SEQ]
        if ids.numel() < 2: break
        ids = ids.unsqueeze(0)
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
        for n, m in targets:
            if m.weight.grad is not None:
                fisher[n] += m.weight.grad.detach()**2
        nseq += 1
    model.zero_grad(set_to_none=True)
    for n in fisher: fisher[n] /= max(nseq, 1)
    print(f"[fisher] accumulated over {nseq} calibration sequences", flush=True)
    return fisher

def restore():
    with torch.no_grad():
        for n, m in targets: m.weight.copy_(orig[n])

def apply_quant(score_fn, p):
    """Ternarize all target weights; keep top-p (per tensor, by score_fn) in fp32."""
    with torch.no_grad():
        for n, m in targets:
            W = orig[n]; Q = absmean_ternary(W)
            if p > 0.0:
                score = score_fn(n, W, Q)            # higher = more important to keep fp32
                k = int(p * W.numel())
                if k > 0:
                    thr = torch.kthvalue(score.flatten(), W.numel() - k).values
                    Q = torch.where(score >= thr, W, Q)
            m.weight.copy_(Q)

def eff_bits(p):  # per quantized weight
    return p*32 + (1-p)*1.58

# --- run ---
restore(); t0 = time.time()
ppl_fp32 = perplexity(model)
print(f"\n[FP32 baseline] PPL = {ppl_fp32:.3f}   ({time.time()-t0:.0f}s)\n", flush=True)

fisher = fisher_diagonal()
scores = {
    "magnitude": lambda n, W, Q: W.abs(),
    "quant_err": lambda n, W, Q: (W - Q)**2,
    "fisher":    lambda n, W, Q: fisher[n] * (W - Q)**2,
}

results = []
restore(); apply_quant(scores["magnitude"], 0.0)
ppl0 = perplexity(model)
results.append(("uniform-ternary", 0.0, eff_bits(0.0), ppl0))
print(f"[p=0.0 uniform absmean ternary] PPL = {ppl0:.3f}  (~{eff_bits(0.0):.2f} bits/w)", flush=True)

for p in [x for x in PROTECT if x > 0]:
    for name, fn in scores.items():
        restore(); apply_quant(fn, p)
        ppl = perplexity(model)
        results.append((name, p, eff_bits(p), ppl))
        print(f"[p={p:.0%} protect by {name:9s}] PPL = {ppl:.3f}  (~{eff_bits(p):.2f} bits/w)", flush=True)

restore()
print("\n================= SUMMARY =================")
print(f"{'condition':22s} {'protect':>8s} {'bits/w':>7s} {'PPL':>9s} {'vs uniform':>11s}")
print(f"{'FP32 (reference)':22s} {'-':>8s} {'32.0':>7s} {ppl_fp32:9.3f} {'-':>11s}")
for name, p, b, ppl in results:
    delta = "" if name == "uniform-ternary" else f"{(ppl-ppl0)/ppl0*100:+.1f}%"
    print(f"{name:22s} {p:8.0%} {b:7.2f} {ppl:9.3f} {delta:>11s}")
print("\nLower PPL = better. If 'fisher' beats 'magnitude' at equal protect%/bits,")
print("the information-geometric SELECTION has measurable value. If not, it does not.")
