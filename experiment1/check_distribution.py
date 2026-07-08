import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_dir = os.path.join(HERE, "detector_ABC")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

df = pd.read_csv(os.path.join(HERE, "stage_A_test.csv"))[["text", "label"]].dropna()

# keep only the MACHINE texts (label 0) — these are the ones we care about forgetting
machine_df = df[df["label"] == 0]
texts = machine_df["text"].tolist()

prob_machine = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        enc = tokenizer(texts[i:i+16], truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, dim=-1)
        prob_machine.extend(p[:, 0].cpu().tolist())

prob_machine = np.array(prob_machine)

# bucket the scores into 10 bins from 0.0 to 1.0
print(f"Detector ABC on Stage A MACHINE texts (n={len(prob_machine)})")
print("P(machine) distribution:\n")
bins = np.arange(0, 1.01, 0.1)
counts, _ = np.histogram(prob_machine, bins=bins)
for i, c in enumerate(counts):
    lo, hi = bins[i], bins[i+1]
    bar = "#" * c
    print(f"{lo:.1f}-{hi:.1f}: {c:4d}  {bar}")

# how many machine texts the detector scores as confidently human
forgotten = (prob_machine < 0.5).sum()
print(f"\nmachine texts scored as human (P<0.5): {forgotten} of {len(prob_machine)} "
      f"({100*forgotten/len(prob_machine):.0f}%)")

# The forgetting gap, stated against the right baseline.
# All numbers from the completed runs (Stage A test set, MachineRec).

detA_at_050   = 0.970  
detABC_at_050 = 0.780  
detABC_recal  = 0.800  

print("Stage A MachineRec — the forgetting story:")
print(f"  Detector A   (trained on A)       @0.50 : {detA_at_050:.3f}   <- baseline")
print(f"  Detector ABC (after B and C)      @0.50 : {detABC_at_050:.3f}")
print(f"  Detector ABC, recalibrated        @0.05 : {detABC_recal:.3f}")
print()
print(f"  total degradation (A -> ABC)            : {detA_at_050 - detABC_at_050:.3f}")
print(f"  recovered by recalibration              : {detABC_recal - detABC_at_050:.3f}")
print(f"  residual, NON-recoverable loss          : {detA_at_050 - detABC_recal:.3f}")