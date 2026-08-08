"""
Two jobs:
  evaluate(model, data) - run the model over a dataset, return a dict of metrics (auroc, human_rec, machine_rec, avg_rec)
  acc / bwt / fwt - reduce the full R matrix to the GEM numbers

Labels: machine = 0, human = 1.

The R matrix has 6 rows and 5 columns:
  row i   = model M_i (trained through stage i); row 0 = untrained baseline
  col j   = performance on the test set of stage j+1
So model M_s scored on stage s (its "just-trained" score) is R[s, s-1].
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data_loading import tokenizer, MAX_LEN
from seeding import get_device

BATCH_SIZE = 64

def evaluate(model, data, max_len=MAX_LEN, return_probs=False):
    """Run `model` over a dataframe with columns text, label. Return metrics.
    With return_probs=True the dict also contains "machine_probs", the
    per-row P(machine) array in the row order of `data` (needed for the
    decomposition analysis and any post-hoc metric)."""
    device = get_device()
    model.eval()
    model.to(device)

    texts = data["text"].tolist()
    labels = data["label"].to_numpy()

    machine_probs = []  # P(machine) for each row (class 0 = machine)
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

    # AUROC with machine as the positive class
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
    """End-state: mean performance of the final model over all test stages."""
    return np.mean(R[5])

def bwt(R):
    """Backward transfer: for each earlier stage, its score after the final
    stage minus its score when it was just trained. Negative = forgetting."""
    return np.mean([R[5, s - 1] - R[s, s - 1] for s in range(1, 5)])

def fwt(R):
    """Forward transfer: for each stage, its score before it was trained
    (the previous model) minus the untrained M0 baseline."""
    return np.mean([R[s - 1, s - 1] - R[0, s - 1] for s in range(2, 6)])