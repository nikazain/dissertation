import glob
import os
import numpy as np
import torch
import data_loading as dl
from config import STAGES
from metrics import evaluate
from train import new_model

PROBS_DIR = "probs"
CKPT_DIR = "checkpoints"

def load_checkpoint(path):
    model = new_model()
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    return model

def main():
    os.makedirs(PROBS_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "stage*_lr*.pt")))
    paths += sorted(glob.glob(os.path.join(CKPT_DIR, "pooled_lr*.pt")))
    if not paths:
        raise SystemExit("no checkpoints found in checkpoints/")
    print("checkpoints to score:")
    for p in paths:
        print("  " + p)

    for path in paths:
        tag = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {tag} ===")
        model = load_checkpoint(path)
        for stage in STAGES:
            out_path = os.path.join(PROBS_DIR, f"val_{tag}_stage{stage}.npz")
            if os.path.exists(out_path):
                print(f"  stage {stage}: already scored, skipping")
                continue
            data = dl.stage_eval(stage, "validation")
            result = evaluate(model, data, return_probs=True)
            np.savez(
                out_path,
                probs=result["machine_probs"],
                labels=data["label"].to_numpy(),
            )
            print(f"  val stage {stage}: auroc {result['auroc']:.4f}"
                  f"  machine_rec {result['machine_rec']:.4f}")
        del model
        torch.cuda.empty_cache()

    print("\nvalidation scoring complete")

if __name__ == "__main__":
    main()