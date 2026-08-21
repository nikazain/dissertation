"""
Truncation experiment, analysis pass (no GPU). Consumes the probabilities
written by truncate_eval.py and produces accuracy as a function of token
budget, following the deployed-threshold protocol of the main threshold
analysis: for each model and each budget L, the threshold is fitted on the
model's own-stage validation set truncated to L, then applied to the test
sets truncated to L. AUROC at each budget is reported alongside, so
ranking loss and decision loss can be told apart.

Run from experiment3/ after the scoring pass has been pulled:
    python truncation_analysis.py
Output: results_truncation.json + printed tables.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

E2 = "."
PROBS = os.path.join(E2, "probs")
LENGTHS = [1024, 512, 256, 128, 64, 32]
GRID = np.arange(0.001, 1.0, 0.001)


def load(path):
    d = np.load(path)
    return d["probs"], d["labels"]


def fit_threshold(probs, labels):
    pred = probs[None, :] > GRID[:, None]
    correct = np.where(labels[None, :] == 0, pred, ~pred)
    return GRID[int(np.argmax(correct.mean(axis=1)))]


def grade(probs, labels, t):
    pred = probs > t
    correct = np.where(labels == 0, pred, ~pred)
    return {
        "accuracy": float(correct.mean()),
        "machine_rec": float(pred[labels == 0].mean()),
        "human_rec": float((~pred)[labels == 1].mean()),
        "auroc": float(roc_auc_score(1 - labels, probs)),
        "threshold": float(t),
    }


def main():
    with open(os.path.join(E2, "results.json")) as f:
        seq = json.load(f)
    with open(os.path.join(E2, "results_pooled.json")) as f:
        pooled = json.load(f)

    models = []
    for s in range(1, 6):
        lr = seq["chosen_lr_per_stage"][f"stage{s}"]
        models.append((f"M{s}", f"stage{s}_lr{lr}", s))
    models.append(("pooled", f"pooled_lr{pooled['chosen_lr']}", None))

    out = {}
    for L in LENGTHS:
        out[L] = {}
        for name, tag, own_stage in models:
            # deployed threshold at this budget
            if own_stage is not None:
                vp, vl = load(os.path.join(
                    PROBS, f"trunc{L}_val_{tag}_stage{own_stage}.npz"))
            else:
                pieces = [load(os.path.join(
                    PROBS, f"trunc{L}_val_{tag}_stage{s}.npz"))
                    for s in range(1, 6)]
                vp = np.concatenate([p for p, _ in pieces])
                vl = np.concatenate([l for _, l in pieces])
            t = fit_threshold(vp, vl)
            out[L][name] = {}
            for s in range(1, 6):
                tp, tl = load(os.path.join(
                    PROBS, f"trunc{L}_{tag}_stage{s}.npz"))
                out[L][name][f"S{s}"] = grade(tp, tl, t)

    with open("results_truncation.json", "w") as f:
        json.dump(out, f, indent=2)

    # summary: mean accuracy over the five stages, per model per budget
    print("\n=== mean test accuracy over stages (deployed threshold per budget) ===")
    print("budget " + "".join(f"  {n:>7s}" for n, _, _ in models))
    for L in LENGTHS:
        row = f"{L:6d}"
        for name, _, _ in models:
            vals = [out[L][name][f"S{s}"]["accuracy"] for s in range(1, 6)]
            row += f"  {np.mean(vals):7.3f}"
        print(row)

    print("\n=== mean test AUROC over stages ===")
    print("budget " + "".join(f"  {n:>7s}" for n, _, _ in models))
    for L in LENGTHS:
        row = f"{L:6d}"
        for name, _, _ in models:
            vals = [out[L][name][f"S{s}"]["auroc"] for s in range(1, 6)]
            row += f"  {np.mean(vals):7.3f}"
        print(row)

    print("\nsaved results_truncation.json (per-stage detail inside)")


if __name__ == "__main__":
    main()