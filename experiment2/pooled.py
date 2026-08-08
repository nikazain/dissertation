"""
Pooled joint-training upper bound: one model trained on all five stages'
(capped) machine rows plus the fixed human sample together, with the same
3-learning-rate sweep as the sequential run. Selection by pooled validation
AUROC (all stages' machine validation rows + the fixed human validation
sample). Every branch is evaluated on all five test sets with per-row
probabilities saved; the winner's checkpoint is saved in fp16.

Run as:  TRAIN_CAP=10000 nohup python -u pooled.py > pooled.log 2>&1 &
(use the same TRAIN_CAP as the sequential run so the bounds are comparable)
"""

import json
import os

import numpy as np
import pandas as pd
import torch

import data_loading as dl
from config import SEED, STAGES
from metrics import evaluate
from seeding import seed_everything
from train import new_model, train_stage

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
RESULTS_PATH = "results_pooled.json"
PROBS_DIR = "probs"
CKPT_DIR = "checkpoints"


def pooled_validation():
    """All stages' machine validation rows + the fixed human validation
    sample. Validation and test are never capped, matching the sequential
    design."""
    machine = dl.MACHINE[dl.MACHINE["mage_split"] == "validation"][["text", "label"]]
    data = pd.concat([machine, dl.human_sample("validation")])
    return data.reset_index(drop=True)


def evaluate_all_stages(model, tag):
    row = []
    for stage in STAGES:
        data = dl.stage_eval(stage, "test")
        result = evaluate(model, data, return_probs=True)
        np.savez(
            os.path.join(PROBS_DIR, f"{tag}_stage{stage}.npz"),
            probs=result["machine_probs"],
            labels=data["label"].to_numpy(),
        )
        row.append({
            k: float(v) for k, v in result.items() if k != "machine_probs"
        })
        print(f"    test stage {stage}: auroc {result['auroc']:.4f}"
              f"  human_rec {result['human_rec']:.4f}"
              f"  machine_rec {result['machine_rec']:.4f}")
    return row


def main():
    seed_everything(SEED)
    os.makedirs(PROBS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_df = dl.pooled_train()
    val_df = pooled_validation()
    print(f"pooled train rows: {len(train_df)}, pooled val rows: {len(val_df)}")

    results = {"learning_rates": LEARNING_RATES, "val_auroc": {}, "rows": {}}
    best_val, best_model, best_lr = -1, None, None

    for lr in LEARNING_RATES:
        print(f"\n=== pooled lr {lr} ===")
        branch = train_stage(new_model(), "pooled", lr,
                             train_df=train_df, val_df=val_df)
        val_auroc = float(evaluate(branch, val_df)["auroc"])
        print(f"  pooled lr {lr}: val auroc {val_auroc:.4f}")

        tag = f"pooled_lr{lr}"
        results["val_auroc"][tag] = val_auroc
        results["rows"][tag] = evaluate_all_stages(branch, tag)

        if val_auroc > best_val:
            best_val, best_model, best_lr = val_auroc, branch, lr

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved {RESULTS_PATH}")

    results["chosen_lr"] = best_lr
    sd = {
        k: (v.half() if v.is_floating_point() else v)
        for k, v in best_model.state_dict().items()
    }
    torch.save(sd, os.path.join(CKPT_DIR, f"pooled_lr{best_lr}.pt"))
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\npooled winner: lr {best_lr} (val auroc {best_val:.4f})")


if __name__ == "__main__":
    main()