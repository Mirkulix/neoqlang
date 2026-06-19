"""
partC: Packed 2-bit ternary kernel — actual DRAM bandwidth savings.

Key insight from RESULTS.md:
  f32-stored quant = 0 energy saving (DRAM bandwidth unchanged).
  Only packed sub-byte storage + a kernel that reads packed bytes saves J/token.

This script validates and benchmarks:
  1. Reference f32-stored matvec (current repo: no savings)
  2. Inline-decoded 2-bit packed matvec (actual savings)
  3. Bandwidth model: theoretical J/token for each

Encoding (matches Rust pack_ternary):
  4 weights per byte, 2 bits each, weight i → byte i//4, bit offset (i%4)*2
  00 = 0,  01 = +1,  11 = -1
"""

import time
import math
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Packing helpers (must match Rust pack_ternary exactly)
# ---------------------------------------------------------------------------

def pack_ternary(w_f32: np.ndarray):
    """Pack f32 ternary weights to 2-bit (00=0, 01=+1, 11=-1). Returns (bytes, alpha)."""
    flat = w_f32.ravel()
    non_zero = np.abs(flat[flat != 0.0])
    alpha = float(non_zero.mean()) if len(non_zero) > 0 else 1.0

    n = len(flat)
    n_bytes = (n + 3) // 4
    packed = bytearray(n_bytes)
    for i, val in enumerate(flat):
        if val > 0.5:
            bits = 0b01
        elif val < -0.5:
            bits = 0b11
        else:
            bits = 0b00
        packed[i // 4] |= bits << ((i % 4) * 2)
    return bytes(packed), alpha


def unpack_ternary(packed: bytes, n_weights: int, alpha: float) -> np.ndarray:
    """Unpack 2-bit packed bytes back to f32 array."""
    out = np.zeros(n_weights, dtype=np.float32)
    for i in range(n_weights):
        bits = (packed[i // 4] >> ((i % 4) * 2)) & 0b11
        if bits == 0b01:
            out[i] = alpha
        elif bits == 0b11:
            out[i] = -alpha
    return out


# ---------------------------------------------------------------------------
# Matvec kernels
# ---------------------------------------------------------------------------

def matvec_f32(W: np.ndarray, x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Reference: f32-stored ternary matvec. Same DRAM load as full f32."""
    return W @ x + bias


def matvec_packed_python(packed: bytes, alpha: float, x: np.ndarray,
                          bias: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """Packed 2-bit matvec. Decodes inline, never materialises f32 weight array."""
    n_bytes_per_row = (in_dim + 3) // 4
    y = bias.copy()
    for j in range(out_dim):
        row = packed[j * n_bytes_per_row:(j + 1) * n_bytes_per_row]
        s = 0.0
        for byte_idx, byte in enumerate(row):
            base = byte_idx * 4
            length = min(in_dim - base, 4)
            for bit_pos in range(length):
                bits = (byte >> (bit_pos * 2)) & 0b11
                if bits == 0b01:
                    s += x[base + bit_pos]
                elif bits == 0b11:
                    s -= x[base + bit_pos]
        y[j] += alpha * s
    return y


def matvec_packed_numpy(packed: bytes, alpha: float, x: np.ndarray,
                         bias: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    """Vectorised packed matvec via numpy — correct for any in_dim.

    Uses flat (dense) indexing matching pack_ternary's sequential layout:
    unpack all out_dim*in_dim weights at once, reshape to (out_dim, in_dim),
    then do a single matrix multiply.  No per-row byte-alignment assumed.
    """
    n_total = out_dim * in_dim
    n_bytes = (n_total + 3) // 4
    packed_np = np.frombuffer(packed[:n_bytes], dtype=np.uint8)

    # Extract 2-bit fields for all 4 slots in every byte
    bits0 = (packed_np) & 0b11          # weights 0,4,8,...
    bits1 = (packed_np >> 2) & 0b11     # weights 1,5,9,...
    bits2 = (packed_np >> 4) & 0b11     # weights 2,6,10,...
    bits3 = (packed_np >> 6) & 0b11     # weights 3,7,11,...

    # Interleave back to natural order and trim to n_total
    all_bits = np.stack([bits0, bits1, bits2, bits3], axis=1).ravel()[:n_total]

    # Map 2-bit codes to ±1: 01→+1, 11(=3 in uint8)→-1, 00→0
    signs = np.where(all_bits == 1, 1.0,
            np.where(all_bits == 3, -1.0, 0.0)).astype(np.float32)

    W_decoded = signs.reshape(out_dim, in_dim) * alpha
    return W_decoded @ x + bias


# ---------------------------------------------------------------------------
# DRAM bandwidth model (from RESULTS.md)
# ---------------------------------------------------------------------------

DRAM_PJ_PER_BYTE = 20.0   # pJ per byte (LPDDR4/DDR4 typical)

def bandwidth_model(out_dim: int, in_dim: int, tokens_per_seq: int = 1000):
    """Theoretical J/token from DRAM weight reads (activations negligible)."""
    w_f32_bytes = out_dim * in_dim * 4
    w_packed_bytes = (out_dim * in_dim + 3) // 4 * 1  # 2 bits/weight → 1/4 of int8

    # For one matmul (one token, weights re-read from DRAM)
    j_f32 = w_f32_bytes * DRAM_PJ_PER_BYTE * 1e-12
    j_packed = w_packed_bytes * DRAM_PJ_PER_BYTE * 1e-12

    return {
        "f32_bytes": w_f32_bytes,
        "packed_bytes": w_packed_bytes,
        "ratio": w_f32_bytes / w_packed_bytes,
        "j_per_tok_f32": j_f32,
        "j_per_tok_packed": j_packed,
        "savings_x": j_f32 / j_packed,
    }


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------

def validate_packed_kernel(out_dim: int = 32, in_dim: int = 64):
    rng = np.random.default_rng(42)
    W = rng.choice([-1.0, 0.0, 1.0], size=(out_dim, in_dim),
                   p=[0.3, 0.4, 0.3]).astype(np.float32)
    x = rng.standard_normal(in_dim).astype(np.float32)
    bias = rng.standard_normal(out_dim).astype(np.float32)

    packed, alpha = pack_ternary(W.ravel())

    y_ref = matvec_f32(W, x, bias)
    y_py = matvec_packed_python(packed, alpha, x, bias, out_dim, in_dim)
    y_np = matvec_packed_numpy(packed, alpha, x, bias, out_dim, in_dim)

    max_err_py = float(np.abs(y_ref - y_py).max())
    max_err_np = float(np.abs(y_ref - y_np).max())

    ok = max_err_py < 1e-4 and max_err_np < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"[validate {out_dim}x{in_dim}]  python={max_err_py:.2e}  numpy={max_err_np:.2e}  {status}")
    return ok


def validate_unaligned(in_dim: int = 7):
    """Validate when in_dim is not a multiple of 4 (boundary padding must be ignored)."""
    out_dim = 3
    rng = np.random.default_rng(7)
    W = rng.choice([-1.0, 0.0, 1.0], size=(out_dim, in_dim)).astype(np.float32)
    x = rng.standard_normal(in_dim).astype(np.float32)
    bias = np.zeros(out_dim, dtype=np.float32)
    packed, alpha = pack_ternary(W.ravel())
    y_ref = matvec_f32(W, x, bias)
    y_np = matvec_packed_numpy(packed, alpha, x, bias, out_dim, in_dim)
    max_err = float(np.abs(y_ref - y_np).max())
    status = "PASS" if max_err < 1e-4 else "FAIL"
    print(f"[validate unaligned in_dim={in_dim}]  max_err={max_err:.2e}  {status}")


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def bench(label: str, fn, n_runs: int = 10):
    # Warmup
    fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = fn()
    elapsed = time.perf_counter() - t0
    ms_per = elapsed / n_runs * 1000
    return ms_per, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("partC: Packed 2-bit Ternary Kernel — Bandwidth Savings")
    print("=" * 62)

    # --- Correctness ---
    print("\n[1] Correctness validation")
    all_ok = True
    all_ok &= validate_packed_kernel(out_dim=8, in_dim=8)    # tiny, fully aligned
    all_ok &= validate_packed_kernel(out_dim=32, in_dim=64)  # small
    all_ok &= validate_packed_kernel(out_dim=64, in_dim=128) # medium
    validate_unaligned(in_dim=7)
    validate_unaligned(in_dim=9)
    if not all_ok:
        print("ERROR: correctness check failed — aborting benchmark")
        return

    # --- Bandwidth model ---
    # GPT-2-scale: 768→768 for attention projections
    out_dim, in_dim = 768, 768
    model = bandwidth_model(out_dim, in_dim)
    print(f"\n[2] DRAM bandwidth model ({out_dim}×{in_dim} weight matrix, 20 pJ/byte)")
    print(f"    f32  weight bytes:   {model['f32_bytes'] / 1024:.1f} KB")
    print(f"    2bit weight bytes:   {model['packed_bytes'] / 1024:.2f} KB")
    print(f"    Size ratio:          {model['ratio']:.1f}×")
    print(f"    J/token f32:         {model['j_per_tok_f32']*1e6:.3f} µJ")
    print(f"    J/token packed:      {model['j_per_tok_packed']*1e6:.3f} µJ")
    print(f"    Theoretical savings: {model['savings_x']:.1f}×")

    # --- Timing ---
    print(f"\n[3] Wall-clock timing (single matvec, {out_dim}×{in_dim})")
    rng = np.random.default_rng(0)
    W = rng.choice([-1.0, 0.0, 1.0], size=(out_dim, in_dim),
                   p=[0.3, 0.4, 0.3]).astype(np.float32)
    x = rng.standard_normal(in_dim).astype(np.float32)
    bias = rng.standard_normal(out_dim).astype(np.float32)
    packed, alpha = pack_ternary(W.ravel())

    ms_f32, _  = bench("f32 numpy",     lambda: matvec_f32(W, x, bias),         n_runs=200)
    ms_np, _   = bench("packed numpy",  lambda: matvec_packed_numpy(packed, alpha, x, bias, out_dim, in_dim), n_runs=50)

    print(f"    f32 numpy matvec:    {ms_f32:.3f} ms/call")
    print(f"    packed numpy matvec: {ms_np:.3f} ms/call")

    # Note: Python-loop packed kernel is for clarity, not speed
    print()
    print("[4] Memory layout")
    zero_frac = float((W == 0).mean())
    nz_frac = 1.0 - zero_frac
    print(f"    Sparsity (zeros):    {zero_frac:.0%}")
    print(f"    Non-zero weights:    {nz_frac:.0%}")
    print(f"    Packed size:         {len(packed)} bytes")
    print(f"    f32 size:            {W.nbytes} bytes")
    print(f"    Actual ratio:        {W.nbytes / len(packed):.1f}×")

    # --- Honest caveat ---
    print()
    print("[5] Honest caveats")
    print("    - Python packed kernel is slower than numpy f32 (Python loop overhead)")
    print("    - Real speedup requires SIMD/Rust kernel reading u8 stream directly")
    print("    - The Rust kernel (ternary_packed_matvec) is implemented in ternary_ops.rs")
    print("    - Run `cargo test packed_kernel_bandwidth_comparison -- --nocapture`")
    print("      to see wall-clock speedup of the actual Rust packed kernel")
    print()
    print("    Conclusion: f32-stored quant = 0 energy saving.")
    print("    Packed 2-bit → 16× less DRAM weight traffic → ~4-10× less J/token")
    print("    (depending on DRAM/compute ratio of the target hardware).")
    print("=" * 62)


if __name__ == "__main__":
    main()
