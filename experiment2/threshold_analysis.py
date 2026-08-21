"""
Experiment 3: re-grade every saved model with real decision thresholds.

Metric: ACCURACY, per Lampos (meeting 7): next to AUROC, the comparable
metric for this binary task is accuracy. Thresholds are fitted on
VALIDATION probabilities by maximising accuracy, then applied to TEST
probabilities. Machine/human recall are kept as secondary diagnostics.

Three regimes, all evaluated on the untouched TEST sets:
  fixed      threshold 0.5
  deployed   one threshold per model, fitted on the validation set of the
             stage it was just trained on (what a practitioner would have)
  refit      threshold re-fitted per (model, test stage) on that stage's
             validation set (the free repair)

Inputs (all already produced):
  ../experiment2/results.json, results_pooled.json
  ../experiment2/probs/{tag}_stage{j}.npz          test probabilities
  ../experiment2/probs/val_{tag}_stage{j}.npz      validation probabilities
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


def grade(probs, labels, t):
    pred_machine = probs > t
    correct = np.where(labels == 0, pred_machine, ~pred_machine)
    return {
        "accuracy": float(correct.mean()),
        "machine_rec": float(pred_machine[labels == 0].mean()),
        "human_rec": float((~pred_machine)[labels == 1].mean()),
        "threshold": float(t),
    }


def fit_threshold(probs, labels):
    """Threshold maximising ACCURACY on the given (validation) set."""
    pred = probs[None, :] > GRID[:, None]
    correct = np.where(labels[None, :] == 0, pred, ~pred)
    return GRID[int(np.argmax(correct.mean(axis=1)))]


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

    out = {"metric": "accuracy", "grid_step": 0.001, "regimes": {}}
    tables = {r: {} for r in ["fixed", "deployed", "refit"]}

    for name, tag, own_stage in models:
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
            tables["fixed"][name][f"S{s}"] = grade(tp, tl, 0.5)
            tables["deployed"][name][f"S{s}"] = grade(tp, tl, t_deploy)
            vps, vls = load(os.path.join(PROBS, f"val_{tag}_stage{s}.npz"))
            tables["refit"][name][f"S{s}"] = grade(tp, tl, fit_threshold(vps, vls))

    out["regimes"] = tables
    with open("results_thresholds.json", "w") as f:
        json.dump(out, f, indent=2)

    for r in ["fixed", "deployed", "refit"]:
        print(f"\n=== {r}: accuracy (machine_rec) ===")
        print("model    " + "".join(f"   S{s}            " for s in range(1, 6)))
        for name, _, _ in models:
            row = f"{name:8s}"
            for s in range(1, 6):
                c = tables[r][name][f"S{s}"]
                row += f"  {c['accuracy']:.3f} ({c['machine_rec']:.3f})"
            print(row)

    print("\nsaved results_thresholds.json")


if __name__ == "__main__":
    main()