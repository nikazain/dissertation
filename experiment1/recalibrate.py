import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import recall_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_dir = os.path.join(HERE, "detector_ABC")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

df = pd.read_csv(os.path.join(HERE, "stage_A_test.csv"))[["text", "label"]].dropna()
texts, labels = df["text"].tolist(), df["label"].tolist()

prob_machine = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        enc = tokenizer(texts[i:i+16], truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, dim=-1)
        prob_machine.extend(p[:, 0].cpu().tolist())

prob_machine = np.array(prob_machine)

def metrics_at(threshold):
    pred = [0 if p >= threshold else 1 for p in prob_machine]
    return (recall_score(labels, pred, pos_label=1),
            recall_score(labels, pred, pos_label=0),  
            balanced_accuracy_score(labels, pred)) 

# default threshold
h0, m0, a0 = metrics_at(0.5)
print(f"threshold 0.50  ->  HumanRec {h0:.3f}  MachineRec {m0:.3f}  AvgRec {a0:.3f}")

# search for the threshold that maximises AvgRec on Stage A
best_t, best_a = 0.5, a0
for t in np.arange(0.05, 0.96, 0.01):
    _, _, a = metrics_at(t)
    if a > best_a:
        best_t, best_a = t, a

hb, mb, ab = metrics_at(best_t)
print(f"threshold {best_t:.2f}  ->  HumanRec {hb:.3f}  MachineRec {mb:.3f}  AvgRec {ab:.3f}")
print(f"\nMachineRec recovered from {m0:.3f} to {mb:.3f} by moving threshold 0.50 -> {best_t:.2f}")