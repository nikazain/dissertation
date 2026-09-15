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
        chosen_lrs = prev["chosen_lr_per_stage"]
        selection_log = prev["selection_log"]
        branches = prev["branches"]
        branches_full = prev["branches_full"]
        done_stage = len(R) - 1
        print(f"resuming: {done_stage} completed stage(s) found in {RESULTS_PATH}")

    if done_stage == 0:
        print("evaluating untrained baseline (M0)")
        row0, full0 = evaluate_all_stages(new_model(), "M0")
        R, R_full = [row0], [full0]
        save(R, R_full, chosen_lrs, selection_log, branches, branches_full, "in progress")

    model = new_model()
    if done_stage > 0:
        lr = chosen_lrs[f"stage{done_stage}"]
        ckpt = torch.load(
            os.path.join(CKPT_DIR, f"seqacc_stage{done_stage}_lr{lr}.pt"),
            map_location="cpu",
        )
        model.load_state_dict(ckpt)
        print(f"loaded checkpoint seqacc_stage{done_stage}_lr{lr}.pt")

    for stage in STAGES:
        if stage <= done_stage:
            continue
        print(f"\n=== stage {stage} ===")
        start_weights = copy.deepcopy(model.state_dict())
        vdata = dl.stage_eval(stage, "validation")
        vlabels = vdata["label"].to_numpy()
        best_sel = -1
        best_model = None
        best_lr = None

        for lr in LEARNING_RATES:
            print(f"\n  training lr {lr}")
            branch = new_model()
            branch.load_state_dict(start_weights)
            branch = train_stage(branch, stage, lr)

            r = evaluate(branch, vdata, return_probs=True)
            t = fit_threshold(r["machine_probs"], vlabels)
            sel = accuracy_at(r["machine_probs"], vlabels, t)
            selection_log[f"stage{stage}_lr{lr}"] = {
                "val_accuracy": sel, "threshold": t,
            }
            print(f"  lr {lr}: threshold {t:.3f}  val accuracy {sel:.4f}")

            print(f"  evaluating lr {lr} on all test stages")
            tag = f"stage{stage}_lr{lr}"
            branches[tag], branches_full[tag] = evaluate_all_stages(branch, tag)

            if sel > best_sel:
                best_sel = sel
                best_model = branch
                best_lr = lr

        print(f"\n  stage {stage} winner: lr {best_lr} (val accuracy {best_sel:.4f})")
        model = best_model
        chosen_lrs[f"stage{stage}"] = best_lr
        save_checkpoint(model, stage, best_lr)
        R.append(branches[f"stage{stage}_lr{best_lr}"])
        R_full.append(branches_full[f"stage{stage}_lr{best_lr}"])
        save(R, R_full, chosen_lrs, selection_log, branches, branches_full, "in progress")

    save(R, R_full, chosen_lrs, selection_log, branches, branches_full, "complete")

    A = np.array(R)
    print(f"\nchosen learning rates: {chosen_lrs}")
    print(f"ACC {acc(A):.4f}   BWT {bwt(A):.4f}   FWT {fwt(A):.4f}")

if __name__ == "__main__":
    main()