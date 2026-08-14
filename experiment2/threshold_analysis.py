"""
Experiment 3: re-grade every saved model with real decision thresholds.

Lampos's point: AUROC assumes a perfect threshold, which never exists in
deployment. This script produces the realistic picture. No training, no GPU.

Three grading regimes, all evaluated on the untouched TEST sets:
  fixed      threshold 0.5 (what the original logs implicitly used)
  deployed   one threshold per model, fitted on the validation set of the
             stage it was just trained on (what a practitioner would have)
  refit      threshold re-fitted per (model, test stage) on that stage's
             validation set (how much is repairable with a sliver of old
             validation data and zero retraining)

Thresholds are always fitted on VALIDATION probabilities and applied to TEST
probabilities, so test data never influences the fit.

Inputs (produced by experiment2):
  ../experiment2/results.json, results_pooled.json   (which lr won where)
  ../experiment2/probs/{tag}_stage{j}.npz            (test probabilities)
  ../experiment2/probs/val_{tag}_stage{j}.npz        (validation, from
                                                      score_validation.py)
Output: results_thresholds.json + printed matrices.
Run from inside experiment3/:  python threshold_analysis.py
"""

import json
import os

import numpy as np

E2 = os.path.join("..", "experiment2")
PROBS = os.path.join(E2, "probs")
GRID = np.arange(0.001, 1.0, 0.001)


def load(path):
    d = np.load(path)
    return d["probs"], d["labels"]


def rec(probs, labels, t):
    pred_machine = probs > t
    machine_rec = float(pred_machine[labels == 0].mean())
    human_rec = float((~pred_machine)[labels == 1].mean())
    return {"machine_rec": machine_rec, "human_rec": human_rec,
            "avg_rec": (machine_rec + human_rec) / 2, "threshold": float(t)}


def fit_threshold(probs, labels):
    """Threshold maximising balanced accuracy (avg_rec) on the given set."""
    pred = probs[None, :] > GRID[:, None]
    m = pred[:, labels == 0].mean(axis=1)
    h = (~pred)[:, labels == 1].mean(axis=1)
    return GRID[int(np.argmax((m + h) / 2))]


def main():
    with open(os.path.join(E2, "results.json")) as f:
        seq = json.load(f)
    with open(os.path.join(E2, "results_pooled.json")) as f:
        pooled = json.load(f)

    models = []  # (name, tag, fitting stage for the deployed regime)
    for s in range(1, 6):
        lr = seq["chosen_lr_per_stage"][f"stage{s}"]
        models.append((f"M{s}", f"stage{s}_lr{lr}", s))
    models.append(("pooled", f"pooled_lr{pooled['chosen_lr']}", None))

    out = {"grid_step": 0.001, "regimes": {}}
    tables = {r: {} for r in ["fixed", "deployed", "refit"]}

    for name, tag, own_stage in models:
        # deployed threshold: fitted once, on the just-trained stage's val
        # (pooled: fitted on all stages' val concatenated)
        if own_stage is not None:
            vp, vl = load(os.path.join(PROBS, f"val_{tag}_stage{own_stage}.npz"))
        else:
            vps, vls = zip(*[load(os.path.join(PROBS, f"val_{tag}_stage{s}.npz"))
                             for s in range(1, 6)])
            vp, vl = np.concatenate(vps), np.concatenate(vls)
        t_deploy = fit_threshold(vp, vl)

        for r in tables:
            tables[r][name] = {}
        for s in range(1, 6):
            tp, tl = load(os.path.join(PROBS, f"{tag}_stage{s}.npz"))
            tables["fixed"][name][f"S{s}"] = rec(tp, tl, 0.5)
            tables["deployed"][name][f"S{s}"] = rec(tp, tl, t_deploy)
            vps, vls = load(os.path.join(PROBS, f"val_{tag}_stage{s}.npz"))
            tables["refit"][name][f"S{s}"] = rec(tp, tl, fit_threshold(vps, vls))

    out["regimes"] = tables
    with open("results_thresholds.json", "w") as f:
        json.dump(out, f, indent=2)

    for r in ["fixed", "deployed", "refit"]:
        print(f"\n=== {r}: avg_rec (machine_rec) ===")
        header = "model    " + "".join(f"   S{s}            " for s in range(1, 6))
        print(header)
        for name, _, _ in models:
            row = f"{name:8s}"
            for s in range(1, 6):
                c = tables[r][name][f"S{s}"]
                row += f"  {c['avg_rec']:.3f} ({c['machine_rec']:.3f})"
            print(row)

    print("\nsaved results_thresholds.json")


if __name__ == "__main__":
    main()