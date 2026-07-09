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
"""

import copy
import json

import numpy as np

import data_loading as dl
from config import SEED, STAGES
from metrics import acc, bwt, evaluate, fwt
from seeding import seed_everything
from train import new_model, train_stage

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
RESULTS_PATH = "results.json"


def evaluate_all_stages(model):
    """Score `model` on every stage's test set. Returns one row of R."""
    row = []
    for stage in STAGES:
        result = evaluate(model, dl.stage_eval(stage, "test"))
        row.append(result["auroc"])
        print(f"    test stage {stage}: auroc {result['auroc']:.4f}"
              f"  human_rec {result['human_rec']:.4f}"
              f"  machine_rec {result['machine_rec']:.4f}")
    return row


def save(R, chosen_lrs, branches, note):
    out = {
        "note": note,
        "learning_rates": LEARNING_RATES,
        "chosen_lr_per_stage": chosen_lrs,
        "R": R,
        "branches": branches,
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

    # Row 0: the untrained model, the FWT baseline.
    print("evaluating untrained baseline (M0)")
    R = [evaluate_all_stages(new_model())]
    chosen_lrs = {}
    branches = {}
    save(R, chosen_lrs, branches, "in progress")

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
            branches[f"stage{stage}_lr{lr}"] = evaluate_all_stages(branch)

            if val_auroc > best_val:
                best_val = val_auroc
                best_model = branch
                best_lr = lr

        print(f"\n  stage {stage} winner: lr {best_lr} (val auroc {best_val:.4f})")
        model = best_model
        chosen_lrs[f"stage{stage}"] = best_lr
        R.append(branches[f"stage{stage}_lr{best_lr}"])
        save(R, chosen_lrs, branches, "in progress")

    save(R, chosen_lrs, branches, "complete")

    A = np.array(R)
    print(f"\nchosen learning rates: {chosen_lrs}")
    print(f"ACC {acc(A):.4f}   BWT {bwt(A):.4f}   FWT {fwt(A):.4f}")


if __name__ == "__main__":
    main()