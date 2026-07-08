import sys
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))

STAGE = sys.argv[1] 

# maps each run to (training csv, model to start from, where to save)
PLAN = {
    "A":("stage_A_train.csv", "allenai/longformer-base-4096","detector_A"),
    "AB":("stage_B_train.csv", "detector_A", "detector_AB"),
    "ABC":("stage_C_train.csv", "detector_AB", "detector_ABC"),
}
train_csv, start_from, save_to = PLAN[STAGE]

MAX_LEN = 512  
EPOCHS = 1 
BATCH = 4  
LR = 2e-5 
SEED = 42

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"=== stage {STAGE}: train on {train_csv}, start from {start_from} ===")
print("device:", device)

# load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(start_from)
model = AutoModelForSequenceClassification.from_pretrained(
    start_from, num_labels=2).to(device)

# load this stage's training data
df = pd.read_csv(os.path.join(HERE, train_csv))
df = df[["text", "label"]].dropna()

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(df).map(tokenize, batched=True)

# training
args = TrainingArguments(
    output_dir=os.path.join(HERE, "tmp_" + save_to),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    seed=SEED,
    logging_steps=50,
    save_strategy="no",         
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,tokenizer=tokenizer)
trainer.train()

# save this detector
out = os.path.join(HERE, save_to)
model.save_pretrained(out)
tokenizer.save_pretrained(out)
print("saved detector to", out)

# evaluate on all three frozen test sets
def evaluate(test_csv):
    t = pd.read_csv(os.path.join(HERE, test_csv))[["text", "label"]].dropna()
    texts, labels = t["text"].tolist(), t["label"].tolist()
    prob_machine = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tokenizer(texts[i:i+16], truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, dim=-1)
            prob_machine.extend(p[:, 0].cpu().tolist())
    pred = [0 if p >= 0.5 else 1 for p in prob_machine]
    return {
        "HumanRec": round(recall_score(labels, pred, pos_label=1), 4),
        "MachineRec": round(recall_score(labels, pred, pos_label=0), 4),
        "AvgRec": round(balanced_accuracy_score(labels, pred), 4),
        "AUROC": round(roc_auc_score([1 if l == 0 else 0 for l in labels], prob_machine), 4),
    }

print("\n=== detector", STAGE, "on each frozen test set ===")
for s in ["A", "B", "C"]:
    print(f"test {s}:", evaluate(f"stage_{s}_test.csv"))