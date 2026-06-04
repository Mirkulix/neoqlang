#!/usr/bin/env python3
"""
Part A: Quality-vs-bits frontier. Does Fisher-protected ternary beat real GPTQ
at comparable bit budgets? Honest PTQ comparison on GPT-2 / WikiText-2.

Conditions:
  FP32 ref | RTN-INT4 | GPTQ-4bit | GPTQ-3bit | GPTQ-2bit(=ternary w/ error comp)
  | absmean-ternary (uniform) | Fisher-protected ternary 5%
GPTQ = own implementation: per-row symmetric int-b, Hessian error feedback (OBQ/GPTQ).
"""
import math, time, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(min(16,(torch.get_num_threads() or 8)))
MODEL="gpt2"; SEQ=512; N_WINDOWS=60; CALIB=24
print(f"[load] {MODEL}", flush=True)
tok=AutoTokenizer.from_pretrained(MODEL)
base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32); base.train(False)

def is_target(n,m):
    return (hasattr(m,"weight") and m.weight is not None and m.weight.dim()==2
            and type(m).__name__ in("Conv1D","Linear")
            and not any(s in n for s in("wte","wpe","lm_head")))
tnames=[n for n,m in base.named_modules() if is_target(n,m)]
def is_conv1d(m): return type(m).__name__=="Conv1D"
print(f"[targets] {len(tnames)} layers", flush=True)
orig={n:dict(base.named_modules())[n].weight.detach().clone() for n in tnames}

print("[data] WikiText-2", flush=True)
test=load_dataset("wikitext","wikitext-2-raw-v1",split="test")
train=load_dataset("wikitext","wikitext-2-raw-v1",split="train")
test_ids=tok("\n\n".join(test["text"]),return_tensors="pt").input_ids[0]
calib_ids=tok("\n\n".join(t for t in train["text"] if len(t)>200)[:200000],return_tensors="pt").input_ids[0]

@torch.no_grad()
def ppl(m):
    m.train(False); nll=0.0; nt=0
    for w in range(N_WINDOWS):
        ids=test_ids[w*SEQ:w*SEQ+SEQ]
        if ids.numel()<2: break
        ids=ids.unsqueeze(0); o=m(ids,labels=ids); n=ids.numel()-1
        nll+=o.loss.item()*n; nt+=n
    return math.exp(nll/nt)

def restore(m):
    d=dict(m.named_modules())
    with torch.no_grad():
        for n in tnames: d[n].weight.copy_(orig[n])

# ---- weight matrix in Linear convention (out,in) ----
def get_W(mod):  return mod.weight.detach().t().clone() if is_conv1d(mod) else mod.weight.detach().clone()
def set_W(mod,W):
    with torch.no_grad(): mod.weight.copy_(W.t() if is_conv1d(mod) else W)

# ---- simple PTQ methods (no Hessian) ----
def rtn_intb(W,bits):
    qmax=2**(bits-1)-1
    s=(W.abs().amax(dim=1,keepdim=True)/qmax).clamp_min(1e-8)
    return torch.clamp(torch.round(W/s),-qmax,qmax)*s
def absmean_ternary(W):
    g=W.abs().mean().clamp_min(1e-8); return torch.clamp(torch.round(W/g),-1,1)*g

# ---- Fisher diagonal (for the protected-ternary condition) ----
def fisher_diag(m):
    d=dict(m.named_modules()); F={n:torch.zeros_like(orig[n]) for n in tnames}; ns=0
    for k in range(CALIB):
        ids=calib_ids[k*SEQ:k*SEQ+SEQ]
        if ids.numel()<2: break
        ids=ids.unsqueeze(0); m.zero_grad(set_to_none=True)
        o=m(ids,labels=ids); o.loss.backward()
        for n in tnames:
            g=d[n].weight.grad
            if g is not None: F[n]+=g.detach()**2
        ns+=1
    m.zero_grad(set_to_none=True)
    for n in F: F[n]/=max(ns,1)
    return F  # in module-native orientation (matches orig[n])

def fisher_protected_ternary(Wnative,Fnative,p=0.05):
    Q=absmean_ternary(Wnative); score=Fnative*(Wnative-Q)**2
    k=int(p*Wnative.numel())
    if k>0:
        thr=torch.kthvalue(score.flatten(),Wnative.numel()-k).values
        Q=torch.where(score>=thr,Wnative,Q)
    return Q

# ---- GPTQ (own impl): per-row symmetric int-b with Hessian error feedback ----
def gptq_quant(W,H,bits,blocksize=128,percdamp=0.01):
    # W:(rows=out,cols=in) Linear convention ; H:(in,in)
    rows,cols=W.shape; W=W.clone().float(); H=H.clone().float()
    qmax=2**(bits-1)-1
    s=(W.abs().amax(dim=1,keepdim=True)/qmax).clamp_min(1e-8)  # per-row static scale
    svec=s.squeeze(1)
    dead=torch.diag(H)==0; H[dead,dead]=1.0; W[:,dead]=0
    damp=percdamp*torch.mean(torch.diag(H)).clamp_min(1e-8)
    idx=torch.arange(cols); H[idx,idx]+=damp
    L=torch.linalg.cholesky(H)
    Hinv=torch.cholesky_inverse(L)
    Hinv=torch.linalg.cholesky(Hinv,upper=True)  # upper-tri
    Q=torch.zeros_like(W)
    for i1 in range(0,cols,blocksize):
        i2=min(i1+blocksize,cols); cnt=i2-i1
        W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); E1=torch.zeros_like(W1)
        Hi=Hinv[i1:i2,i1:i2]
        for i in range(cnt):
            w=W1[:,i]; d=Hi[i,i]
            q=torch.clamp(torch.round(w/svec),-qmax,qmax)*svec
            Q1[:,i]=q; err=(w-q)/d
            W1[:,i:]-=err.unsqueeze(1)*Hi[i,i:].unsqueeze(0); E1[:,i]=err
        Q[:,i1:i2]=Q1
        if i2<cols: W[:,i2:]-=E1@Hinv[i1:i2,i2:]
    return Q

def collect_hessians(m):
    d=dict(m.named_modules()); H={}; cnt={}
    def mk(n):
        def hook(mod,inp):
            x=inp[0].detach().reshape(-1,inp[0].shape[-1]).float()  # (tokens,in)
            if n not in H: H[n]=torch.zeros(x.shape[1],x.shape[1]); cnt[n]=0
            H[n]+=x.t()@x; cnt[n]+=x.shape[0]
        return hook
    hs=[d[n].register_forward_pre_hook(mk(n)) for n in tnames]
    with torch.no_grad():
        for k in range(CALIB):
            ids=calib_ids[k*SEQ:k*SEQ+SEQ]
            if ids.numel()<2: break
            m(ids.unsqueeze(0))
    for h in hs: h.remove()
    for n in H: H[n]*=2.0/max(cnt[n],1)
    return H

def apply_method(m,fn_native=None,fn_gptq=None,H=None,bits=None):
    d=dict(m.named_modules())
    with torch.no_grad():
        for n in tnames:
            mod=d[n]
            if fn_gptq is not None:
                W=get_W(mod); Q=gptq_quant(W,H[n],bits); set_W(mod,Q)
            else:
                mod.weight.copy_(fn_native(n,orig[n]))

# ===================== RUN =====================
t0=time.time(); restore(base)
p_fp32=ppl(base); print(f"\n[FP32] PPL={p_fp32:.3f} ({time.time()-t0:.0f}s)",flush=True)

print("[gptq] collecting Hessians...",flush=True); t=time.time()
H=collect_hessians(base); print(f"  done ({time.time()-t:.0f}s)",flush=True)
print("[fisher] diagonal...",flush=True); F=fisher_diag(base)

res=[]
# RTN int4
restore(base); apply_method(base,fn_native=lambda n,W:rtn_intb(W if False else orig[n],4) if False else rtn_intb(orig[n],4))
res.append(("RTN-int4",4.0,ppl(base))); print(f"  RTN-int4 PPL={res[-1][2]:.3f}",flush=True)
# GPTQ 4/3/2
for b in (4,3,2):
    restore(base); apply_method(base,fn_gptq=True,H=H,bits=b)
    nm=f"GPTQ-{b}bit"+(" (ternary)" if b==2 else "")
    res.append((nm,float(b),ppl(base))); print(f"  {nm} PPL={res[-1][2]:.3f}",flush=True)
# absmean ternary
restore(base); apply_method(base,fn_native=lambda n,W:absmean_ternary(orig[n]))
res.append(("absmean-ternary",1.58,ppl(base))); print(f"  absmean-ternary PPL={res[-1][2]:.3f}",flush=True)
# fisher-protected ternary 5%
restore(base); apply_method(base,fn_native=lambda n,W:fisher_protected_ternary(orig[n],F[n],0.05))
res.append(("fisher-ternary-5%",3.10,ppl(base))); print(f"  fisher-ternary-5% PPL={res[-1][2]:.3f}",flush=True)
restore(base)

print("\n================ QUALITY vs BITS ================")
print(f"{'method':22s}{'~bits/w':>9s}{'PPL':>11s}{'vs FP32':>10s}")
print(f"{'FP32 (ref)':22s}{'32.0':>9s}{p_fp32:11.3f}{'1.00x':>10s}")
for nm,b,p in sorted(res,key=lambda r:r[1]):
    print(f"{nm:22s}{b:9.2f}{p:11.3f}{p/p_fp32:9.2f}x")
print("\nFrontier = lowest PPL at each bit level. If ternary variants are far above")
print("the GPTQ points, GPTQ dominates the quality-per-bit frontier (expected).")
