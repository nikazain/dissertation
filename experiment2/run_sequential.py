"""
At each stage, trains three models from the current winner (one per learning
rate), picks the best by that stage's validation AUROC, and carries it
forward. Losing branches are still evaluated on all five test sets before
being discarded, which gives the learning-rate-vs-forgetting view.

Fills the 6x5 matrix R (the winner at each stage):
    row 0    untrained model (the M0 baseline, needed for FWT)
    row i    the model after training through stage i
    col j    performance on the test set of stage j+1

Saves results.json after every stage, so a crash costs one stage, not the run.

Persistence for later analysis (all inference-only follow-ups depend on these):
    probs/{tag}_stage{j}.npz   per-row P(machine) + labels for every test eval
    checkpoints/stage{s}_lr{lr}.pt   each stage winner, fp16 state_dict
    results.json also stores all four metrics per cell (R_full, branches_full)
"""

import copy
import json
import os

import numpy as np
import torch

import data_loading as dl
from config import SEED, STAGES
from metrics import acc, bwt, evaluate, fwt
from seeding import seed_everything
from train import new_model, train_stage

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
RESULTS_PATH = "results.json"
PROBS_DIR = "probs"
CKPT_DIR = "checkpoints"


def evaluate_all_stages(model, tag):
    """Score `model` on every stage's test set. Saves per-row probabilities
    under `tag`. Returns (auroc_row, full_metrics_row)."""
    row, full_row = [], []
    for stage in STAGES:
        data = dl.stage_eval(stage, "test")
        result = evaluate(model, data, return_probs=True)
        np.savez(
            os.path.join(PROBS_DIR, f"{tag}_stage{stage}.npz"),
            probs=result["machine_probs"],
            labels=data["label"].to_numpy(),
        )
        row.append(float(result["auroc"]))
        full_row.append({
            k: float(v) for k, v in result.items() if k != "machine_probs"
        })
        print(f"    test stage {stage}: auroc {result['auroc']:.4f}"
              f"  human_rec {result['human_rec']:.4f}"
              f"  machine_rec {result['machine_rec']:.4f}")
    return row, full_row


def save_checkpoint(model, stage, lr):
    """fp16 state_dict (floats halved, integer buffers untouched).
    Loading back into a fresh fp32 model works directly:
    new_model().load_state_dict(torch.load(path)) casts on copy."""
    sd = {
        k: (v.half() if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    torch.save(sd, os.path.join(CKPT_DIR, f"stage{stage}_lr{lr}.pt"))


def save(R, R_full, chosen_lrs, branches, branches_full, note):
    out = {
        "note": note,
        "learning_rates": LEARNING_RATES,
        "chosen_lr_per_stage": chosen_lrs,
        "R": R,
        "R_full": R_full,
        "branches": branches,
        "branches_full": branches_full,
    }
    if len(R) == 6:
        A = np.array(R)
        out["acc"] = float(acc(A))
        out["bwt"] = float(bwt(A))
        out["fwt"] = float(fwt(A))
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {RESULTS_PATH} ({len(R)} rows)")


def main():
    seed_everything(SEED)
    os.makedirs(PROBS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Row 0: the untrained model, the FWT baseline.
    print("evaluating untrained baseline (M0)")
    row0, full0 = evaluate_all_stages(new_model(), "M0")
    R, R_full = [row0], [full0]
    chosen_lrs = {}
    branches = {}
    branches_full = {}
    save(R, R_full, chosen_lrs, branches, branches_full, "in progress")

    model = new_model()  # the current winner, carried between stages

    for stage in STAGES:
        print(f"\n=== stage {stage} ===")
        start_weights = copy.deepcopy(model.state_dict())
        val_data = dl.stage_eval(stage, "validation")

        best_val = -1
        best_model = None
        best_lr = None

        for lr in LEARNING_RATES:
            print(f"\n  training lr {lr}")
            branch = new_model()
            branch.load_state_dict(start_weights)  # every branch starts here
            branch = train_stage(branch, stage, lr)

            val_auroc = evaluate(branch, val_data)["auroc"]
            print(f"  lr {lr}: val auroc {val_auroc:.4f}")

            print(f"  evaluating lr {lr} on all test stages")
            tag = f"stage{stage}_lr{lr}"
            branches[tag], branches_full[tag] = evaluate_all_stages(branch, tag)

            if val_auroc > best_val:
                best_val = val_auroc
                best_model = branch
                best_lr = lr

        print(f"\n  stage {stage} winner: lr {best_lr} (val auroc {best_val:.4f})")
        model = best_model
        chosen_lrs[f"stage{stage}"] = best_lr
        save_checkpoint(model, stage, best_lr)
        R.append(branches[f"stage{stage}_lr{best_lr}"])
        R_full.append(branches_full[f"stage{stage}_lr{best_lr}"])
        save(R, R_full, chosen_lrs, branches, branches_full, "in progress")

    save(R, R_full, chosen_lrs, branches, branches_full, "complete")

    A = np.array(R)
    print(f"\nchosen learning rates: {chosen_lrs}")
    print(f"ACC {acc(A):.4f}   BWT {bwt(A):.4f}   FWT {fwt(A):.4f}")


if __name__ == "__main__":
    main()