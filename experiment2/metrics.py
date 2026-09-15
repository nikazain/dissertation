import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from data_loading import tokenizer, MAX_LEN
from seeding import get_device

BATCH_SIZE = 64

def evaluate(model, data, max_len=MAX_LEN, return_probs=False):
    device = get_device()
    model.eval()
    model.to(device)

    texts = data["text"].tolist()
    labels = data["label"].to_numpy()

    machine_probs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)
        machine_probs.extend(probs[:, 0].cpu().numpy())
    machine_probs = np.array(machine_probs)
    predicted_machine = machine_probs > 0.5
    auroc = roc_auc_score(labels == 0, machine_probs)
    is_machine = labels == 0
    is_human = labels == 1
    machine_rec = predicted_machine[is_machine].mean()
    human_rec = (~predicted_machine[is_human]).mean()
    avg_rec = (machine_rec + human_rec) / 2
    out = {
        "auroc": auroc,
        "human_rec": human_rec,
        "machine_rec": machine_rec,
        "avg_rec": avg_rec,
    }
    if return_probs:
        out["machine_probs"] = machine_probs
    return out

def acc(R):
    return np.mean(R[5])

def bwt(R):
    return np.mean([R[5, s - 1] - R[s, s - 1] for s in range(1, 5)])

def fwt(R):
    return np.mean([R[s - 1, s - 1] - R[0, s - 1] for s in range(2, 6)])