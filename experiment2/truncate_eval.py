import glob
import os
import numpy as np
import torch
from transformers import AutoTokenizer
import data_loading as dl
from config import STAGES
from train import new_model

LENGTHS = [1024, 512, 256, 128, 64, 32]
BATCH = 64
PROBS_DIR = "probs"
CKPT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path):
    model = new_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(DEVICE).eval()
    return model

@torch.no_grad()
def score(model, tokenizer, texts, max_len):
    probs = []
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(texts[i:i + BATCH], truncation=True,
                        max_length=max_len, padding="max_length",
                        return_tensors="pt").to(DEVICE)
        gam = torch.zeros_like(enc["attention_mask"])
        gam[:, 0] = 1
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=DEVICE == "cuda"):
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"],
                           global_attention_mask=gam).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 0]  # P(machine), label 0
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)

def maybe_score(model, tokenizer, data, out_path, max_len):
    if os.path.exists(out_path):
        print(f"    exists, skipping: {os.path.basename(out_path)}")
        return
    texts = data["text"].tolist()
    labels = data["label"].to_numpy()
    probs = score(model, tokenizer, texts, max_len)
    np.savez(out_path, probs=probs, labels=labels)
    auroc_proxy = (probs[labels == 0].mean(), probs[labels == 1].mean())
    print(f"    saved {os.path.basename(out_path)}  "
          f"mean p(machine): machine {auroc_proxy[0]:.3f} human {auroc_proxy[1]:.3f}")

def main():
    os.makedirs(PROBS_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    paths = []
    import json
    with open("results.json") as f:
        seq = json.load(f)
    for s in range(1, 6):
        lr = seq["chosen_lr_per_stage"][f"stage{s}"]
        paths.append((f"stage{s}_lr{lr}", s))
    with open("results_pooled.json") as f:
        pooled = json.load(f)
    paths.append((f"pooled_lr{pooled['chosen_lr']}", None))

    for tag, own_stage in paths:
        ckpt = os.path.join(CKPT_DIR, f"{tag}.pt")
        print(f"\n=== {tag} ===")
        model = load_checkpoint(ckpt)
        for L in LENGTHS:
            print(f"  budget {L} tokens")
            # test sets, all stages
            for s in STAGES:
                out = os.path.join(PROBS_DIR, f"trunc{L}_{tag}_stage{s}.npz")
                maybe_score(model, tokenizer, dl.stage_eval(s, "test"), out, L)
            # validation for threshold fitting: own stage, or all for pooled
            val_stages = [own_stage] if own_stage is not None else list(STAGES)
            for s in val_stages:
                out = os.path.join(PROBS_DIR, f"trunc{L}_val_{tag}_stage{s}.npz")
                maybe_score(model, tokenizer, dl.stage_eval(s, "validation"), out, L)
        del model
        torch.cuda.empty_cache()

    print("\ntruncation scoring complete")

if __name__ == "__main__":
    main()