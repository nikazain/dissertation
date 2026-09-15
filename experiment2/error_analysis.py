"""
Error-direction analysis: when the detector makes a mistake, is it
human text flagged as AI or AI text passed as human ?
"""

import json
import os

import numpy as np

PROBS = "probs"
LENGTHS = [1024, 512, 256, 128, 64, 32]


def counts(tag, stage):
    l = np.load(os.path.join(PROBS, f"trunc1024_{tag}_stage{stage}.npz"))["labels"]
    return int((l == 0).sum()), int((l == 1).sum())  # machine, human


def split_errors(cell, n_m, n_h):
    fn = (1 - cell["machine_rec"]) * n_m      # AI passed as human
    fp = (1 - cell["human_rec"]) * n_h        # human flagged as AI (dangerous)
    prec_m = cell["machine_rec"] * n_m / max(cell["machine_rec"] * n_m + fp, 1e-9)
    f1_m = 2 * prec_m * cell["machine_rec"] / max(prec_m + cell["machine_rec"], 1e-9)
    rec_h = cell["human_rec"]
    prec_h = rec_h * n_h / max(rec_h * n_h + fn, 1e-9)
    f1_h = 2 * prec_h * rec_h / max(prec_h + rec_h, 1e-9)
    return {
        "fp_human_as_ai": round(fp), "fn_ai_as_human": round(fn),
        "share_dangerous": fp / max(fp + fn, 1e-9),
        "macro_f1": (f1_m + f1_h) / 2,
    }


def main():
    with open("results.json") as f:
        seq = json.load(f)
    with open("results_pooled.json") as f:
        pooled = json.load(f)
    tags = {f"M{s}": f"stage{s}_lr{seq['chosen_lr_per_stage'][f'stage{s}']}"
            for s in range(1, 6)}
    tags["pooled"] = f"pooled_lr{pooled['chosen_lr']}"

    out = {"deployed": {}, "truncation": {}}

    with open("results_thresholds.json") as f:
        dep = json.load(f)["regimes"]["deployed"]
    print("=== deployed cutoffs: error composition ===")
    print("model  stage  hum->AI  AI->hum  dangerous%  macroF1")
    for m, tag in tags.items():
        out["deployed"][m] = {}
        for s in range(1, 6):
            n_m, n_h = counts(tag, s)
            e = split_errors(dep[m][f"S{s}"], n_m, n_h)
            out["deployed"][m][f"S{s}"] = e
            print(f"{m:>6}   S{s}   {e['fp_human_as_ai']:>6}  {e['fn_ai_as_human']:>7}"
                  f"     {100*e['share_dangerous']:5.1f}%   {e['macro_f1']:.3f}")

    with open("results_truncation.json") as f:
        tr = json.load(f)
    print("\n=== truncation: error composition, mean over stages ===")
    print("budget  model  hum->AI  AI->hum  dangerous%  macroF1")
    for L in LENGTHS:
        out["truncation"][L] = {}
        for m, tag in tags.items():
            es = []
            for s in range(1, 6):
                n_m, n_h = counts(tag, s)
                es.append(split_errors(tr[str(L)][m][f"S{s}"], n_m, n_h))
            agg = {
                "fp_human_as_ai": sum(e["fp_human_as_ai"] for e in es),
                "fn_ai_as_human": sum(e["fn_ai_as_human"] for e in es),
                "macro_f1": float(np.mean([e["macro_f1"] for e in es])),
            }
            agg["share_dangerous"] = agg["fp_human_as_ai"] / max(
                agg["fp_human_as_ai"] + agg["fn_ai_as_human"], 1e-9)
            out["truncation"][L][m] = agg
            print(f"{L:>6}  {m:>5}  {agg['fp_human_as_ai']:>6}  {agg['fn_ai_as_human']:>7}"
                  f"     {100*agg['share_dangerous']:5.1f}%   {agg['macro_f1']:.3f}")

    with open("results_errors.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results_errors.json")


if __name__ == "__main__":
    main()