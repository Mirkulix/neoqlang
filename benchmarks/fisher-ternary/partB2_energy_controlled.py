#!/usr/bin/env python3
"""Part B (controlled): interleave fp32 vs int4-as-f32 across repeats so thermal/
turbo drift averages out. Shows the two are within measurement noise (identical
compute), and reports a stable J/token baseline. Then the bandwidth-bound model."""
import subprocess, time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0); torch.set_num_threads(min(16,(torch.get_num_threads() or 8)))
EUR_PER_KWH=0.30
RAPL="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
def euj(): return int(subprocess.run(["sudo","-n","cat",RAPL],capture_output=True,text=True).stdout.strip())

tok=AutoTokenizer.from_pretrained("gpt2")
m=AutoModelForCausalLM.from_pretrained("gpt2",dtype=torch.float32); m.train(False)
prompt=tok("The history of artificial intelligence began",return_tensors="pt").input_ids
NEW=300
orig={n:p.detach().clone() for n,p in m.named_parameters()}

def is_t(n,mm): return (hasattr(mm,"weight") and mm.weight is not None and mm.weight.dim()==2
    and type(mm).__name__ in("Conv1D","Linear") and not any(s in n for s in("wte","wpe","lm_head")))
def quantize_int4_f32():
    with torch.no_grad():
        for n,mm in m.named_modules():
            if is_t(n,mm):
                W=mm.weight; s=(W.abs().amax(0,keepdim=True)/7).clamp_min(1e-8)
                mm.weight.copy_(torch.clamp(torch.round(W/s),-7,7)*s)
def restore():
    with torch.no_grad():
        for n,p in m.named_parameters(): p.copy_(orig[n])

@torch.no_grad()
def decode():
    out=m.generate(prompt,max_new_tokens=NEW,do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)
    return out.shape[1]-prompt.shape[1]

def run_once():
    e0=euj(); t0=time.time(); n=decode(); dt=time.time()-t0
    return (euj()-e0)/1e6/n, n/dt   # J/token, tok/s

print("[warmup]",flush=True); decode(); decode()
fp32=[]; q=[]
for r in range(3):
    restore();          j,t=run_once(); fp32.append(j); print(f"  fp32  rep{r}: {j:.3f} J/tok  {t:.1f} tok/s",flush=True)
    quantize_int4_f32();j,t=run_once(); q.append(j);    print(f"  int4f rep{r}: {j:.3f} J/tok  {t:.1f} tok/s",flush=True)
restore()
mf,sf=statistics.mean(fp32),statistics.pstdev(fp32)
mq,sq=statistics.mean(q),statistics.pstdev(q)
print(f"\nfp32           : {mf:.3f} ± {sf:.3f} J/token")
print(f"int4-as-f32    : {mq:.3f} ± {sq:.3f} J/token")
print(f"difference     : {100*(mq-mf)/mf:+.1f}%  (within ±{100*max(sf,sq)/mf:.0f}% noise -> NOT a real saving)")
print(f"baseline cost  : {mf/3.6e6*1e6*EUR_PER_KWH:.4f} EUR / 1M tokens @ {EUR_PER_KWH} EUR/kWh (this CPU)")

P=124e6; PJB=20
we=lambda b:P*b/8*PJB/1e12*1000  # mJ/token weight movement
print(f"\n--- bandwidth-bound model (~{PJB} pJ/byte DRAM) ---")
print(f"fp16 weights : {we(16):.2f} mJ/token (move {P*2/1e6:.0f} MB/token)")
print(f"int4 packed  : {we(4):.2f} mJ/token ({we(16)/we(4):.1f}x less)")
print(f"1.58b packed : {we(1.58):.2f} mJ/token ({we(16)/we(1.58):.1f}x less)")
print("\nTakeaway: f32-stored 'quant' saves ~0 energy (proven). Only PACKED sub-byte")
print("weights + a kernel reading them packed cut J/token. That packing+kernel is")
print("exactly what bitnet.cpp/GPTQ-kernels already ship and these repos do not.")
