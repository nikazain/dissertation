"""
Greedy rerun with ACCURACY selection: identical to the original sequential
run except the winner at each stage is chosen by decision accuracy on the
CURRENT stage's validation set, at a threshold fitted there by maximising
accuracy. Resumable.

Outputs:
    results_sequential_acc.json
    probs/seqacc_{tag}_stage{j}.npz
    checkpoints/seqacc_stage{s}_lr{lr}.pt
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

if torch.cuda.is_available():
    _reserve = torch.empty(20 * 1024**3 // 4, dtype=torch.float32, device="cuda")
    del _reserve

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
RESULTS_PATH = "results_sequential_acc.json"
PROBS_DIR = "probs"
CKPT_DIR = "checkpoints"
GRID = np.arange(0.001, 1.0, 0.001)


def fit_threshold(probs, labels):
    pred = probs[None, :] > GRID[:, None]
    correct = np.where(labels[None, :] == 0, pred, ~pred)
    return float(GRID[int(np.argmax(correct.mean(axis=1)))])


def accuracy_at(probs, labels, t):
    pred = probs > t
    return float(np.where(labels == 0, pred, ~pred).mean())


def evaluate_all_stages(model, tag):
    row, full_row = [], []
    for stage in STAGES:
        data = dl.stage_eval(stage, "test")
        result = evaluate(model, data, return_probs=True)
        np.savez(
            os.path.join(PROBS_DIR, f"seqacc_{tag}_stage{stage}.npz"),
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
    sd = {
        k: (v.half() if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    torch.save(sd, os.path.join(CKPT_DIR, f"seqacc_stage{stage}_lr{lr}.pt"))


def save(R, R_full, chosen_lrs, selection_log, branches, branches_full, note):
    out = {
        "note": note,
        "selection": "validation ACCURACY on current stage, threshold fitted "
                     "on current stage validation",
        "learning_rates": LEARNING_RATES,
        "chosen_lr_per_stage": chosen_lrs,
        "selection_log": selection_log,
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

    R, R_full = [], []
    chosen_lrs, selection_log = {}, {}
    branches, branches_full = {}, {}
    done_stage = 0

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            prev = json.load(f)
        R = prev["R"]
        R_full = prev["R_full"]