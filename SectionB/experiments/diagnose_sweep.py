"""Steps 2-4 — L1 (cand_n), L2 (dense-union M), L4 ((dense_w, β) grid) sweeps.

All over the committed artifacts via the shared Lab harness; pure numpy, seconds.
Prints full-set NDCG@10 for each config. CV-gating is in diagnose_cv.py.

Run:  conda run -n DataLab-ProjectA python experiments/diagnose_sweep.py
"""
from __future__ import annotations

from _lab import Lab


def main() -> None:
    lab = Lab()
    base = lab.score_config()
    print(f"baseline (0.8, 0.15, 300) = {base:.4f}\n")

    print("== L1: widen cand_n (dense_w=0.8, β=0.15) ==")
    for cn in (300, 500, 750, 1000, 1500):
        print(f"  cand_n={cn:<5} = {lab.score_config(cand_n=cn):.4f}")

    print("\n== L2: global-dense union top-M (cand_n=300, dense_w=0.8, β=0.15) ==")
    for m in (0, 25, 50, 100, 200, 500):
        print(f"  dense_union_m={m:<4} = {lab.score_config(dense_union_m=m):.4f}")

    print("\n== L4: (dense_w, β) grid (cand_n=300) ==")
    betas = (0.0, 0.05, 0.10, 0.15, 0.20)
    print("  dense_w \\ β " + "".join(f"{b:>8}" for b in betas))
    best = (base, 0.8, 0.15)
    for dw in (0.7, 0.8, 0.85, 0.9, 0.95, 1.0):
        cells = []
        for b in betas:
            v = lab.score_config(dense_w=dw, beta=b)
            cells.append(v)
            if v > best[0]:
                best = (v, dw, b)
        print(f"  dw={dw:<5}    " + "".join(f"{c:>8.4f}" for c in cells))
    print(f"\nbest full-set: {best[0]:.4f} at dense_w={best[1]}, β={best[2]} "
          f"(Δ vs baseline {best[0]-base:+.4f}) — CV-gate before trusting.")


if __name__ == "__main__":
    main()
