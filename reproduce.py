import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import recall_score, roc_auc_score, balanced_accuracy_score

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("running on:", device)

test = load_dataset("yaful/MAGE")["test"].to_pandas()
mage_domains = ["cmv", "eli5", "tldr", "xsum", "wp",
                "roct", "hswag", "yelp", "squad", "sci_gen"]
starts = tuple(d + "_" for d in mage_domains)
part1 = test[test["src"].str.startswith(starts)]
sample = part1.groupby("label").sample(n=500, random_state=42)
texts = sample["text"].tolist()
labels = sample["label"].tolist()
model_name = "yaful/MAGE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()
print("model's label mapping:", model.config.id2label)

prob_machine = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        enc = tokenizer(batch, truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)
        prob_machine.extend(p[:, 0].cpu().tolist())
        print(f"{i+len(batch)}/{len(texts)}", end="\r")

pred = [0 if p >= 0.5 else 1 for p in prob_machine]
human_rec = recall_score(labels, pred, pos_label=1)
machine_rec = recall_score(labels, pred, pos_label=0)
avg_rec = balanced_accuracy_score(labels, pred)
auroc = roc_auc_score([1 if l == 0 else 0 for l in labels], prob_machine)

print("\nHumanRec:  ", round(human_rec, 4))
print("MachineRec:", round(machine_rec, 4))
print("AvgRec:    ", round(avg_rec, 4))
print("AUROC:     ", round(auroc, 4))