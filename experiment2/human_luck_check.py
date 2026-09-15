"""
Human luck check 
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import data_loading as dl
from config import HUMAN_SIZES, TRAIN_CAP
from train import new_model

N_FRESH = HUMAN_SIZES["test"]
BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fresh_humans():
    h = dl.HUMAN
    pool = h[(h["mage_split"] == "test") & (h["shuffle_rank"] >= HUMAN_SIZES["test"])]
    source = "test split, positions beyond the fixed sample"
    if len(pool) < N_FRESH:
        cap = TRAIN_CAP if TRAIN_CAP is not None else HUMAN_SIZES["train"]
        pool = h[(h["mage_split"] == "train") & (h["shuffle_rank"] >= cap)]
        source = f"train split, positions never used for training (rank >= {cap})"
    pool = pool.sort_values("shuffle_rank").head(N_FRESH)
    assert len(pool) == N_FRESH, f"only {len(pool)} fresh humans available"
    print(f"fresh humans: {len(pool)} from {source}")
    return pool["text"].tolist(), source


def load_checkpoint(path):
    model = new_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(DEVICE).eval()


@torch.no_grad()
def score(model, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = dl.tokenize(texts[i:i + BATCH]).to(DEVICE)
        gam = torch.zeros_like(enc["attention_mask"])
        gam[:, 0] = 1
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=DEVICE == "cuda"):
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"],
                           global_attention_mask=gam).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 0].cpu().numpy())
    return np.concatenate(out)  # P(machine)


def grade(p_machine, p_human, t):
    pred_m, pred_h = p_machine > t, p_human > t
    return {
        "accuracy": float(np.concatenate([pred_m, ~pred_h]).mean()),
        "machine_rec": float(pred_m.mean()),
        "human_rec": float((~pred_h).mean()),
        "auroc": float(roc_auc_score(
            np.r_[np.ones(len(p_machine)), np.zeros(len(p_human))],
            np.r_[p_machine, p_human])),
    }


def main():
    with open("results.json") as f:
        seq = json.load(f)
    with open("results_pooled.json") as f:
        pooled = json.load(f)
    with open("results_thresholds.json") as f:
        deployed = json.load(f)["regimes"]["deployed"]

    models = [
        ("M5", f"stage5_lr{seq['chosen_lr_per_stage']['stage5']}"),
        ("pooled", f"pooled_lr{pooled['chosen_lr']}"),
    ]

    texts, source = fresh_humans()
    out = {"fresh_pool": source, "n_fresh": N_FRESH, "models": {}}

    for name, tag in models:
        print(f"\n=== {name} ({tag}) ===")
        model = load_checkpoint(os.path.join("checkpoints", f"{tag}.pt"))
        p_fresh = score(model, texts)
        del model
        torch.cuda.empty_cache()

        out["models"][name] = {}
        print("stage   acc fixed  acc fresh   auroc fixed  auroc fresh   hum_rec fixed  hum_rec fresh")
        for j in range(1, 6):
            d = np.load(f"probs/trunc1024_{tag}_stage{j}.npz")
            p, l = d["probs"], d["labels"]
            t = deployed[name][f"S{j}"]["threshold"]
            fixed = grade(p[l == 0], p[l == 1], t)
            fresh = grade(p[l == 0], p_fresh, t)
            out["models"][name][f"S{j}"] = {"threshold": t, "fixed": fixed, "fresh": fresh}
            print(f"  S{j}     {fixed['accuracy']:.3f}      {fresh['accuracy']:.3f}"
                  f"       {fixed['auroc']:.3f}        {fresh['auroc']:.3f}"
                  f"        {fixed['human_rec']:.3f}          {fresh['human_rec']:.3f}")

    with open("results_human_luck.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results_human_luck.json")


if __name__ == "__main__":
    main()