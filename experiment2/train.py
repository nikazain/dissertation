"""
RUN THIS (via a small script) - trains one model on one stage at one lr.

Also defines new_model(): a fresh Longformer detector (used as the M0
starting point and for the pooled upper bound).

    from train import train_stage, new_model
    model = train_stage(new_model(), stage=1, lr=3e-5)

Smoke test on a small chunk (confirms the loop runs before spending compute):
    model = train_stage(new_model(), stage=1, lr=3e-5, limit=200)

Keeps the epoch with the best validation AUROC. Validation uses the
CURRENT stage's validation set only, so the naive sequential baseline is
not accidentally helped to resist forgetting.
"""

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import data_loading as dl
from data_loading import MODEL_NAME, MAX_LEN, tokenizer
from metrics import evaluate
from seeding import get_device

EPOCHS = 5
BATCH_SIZE = 16
WARMUP_RATIO = 0.06


def new_model():
    """A fresh Longformer with a 2-class classification head."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def train_stage(model, stage, lr, limit=None, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    device = get_device()
    model.to(device)

    train_data = dl.stage_train(stage)
    val_data = dl.stage_eval(stage, "validation")

    # Smoke-test mode: use only a small chunk. Shuffle first so the chunk
    # contains both machine and human rows (otherwise validation AUROC has
    # only one class and errors).
    if limit is not None:
        train_data = train_data.sample(frac=1, random_state=0).head(limit)
        val_data = val_data.sample(frac=1, random_state=0).head(limit)

    enc = tokenizer(
        train_data["text"].tolist(),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.tensor(train_data["label"].to_numpy())
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01
    )
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps
    )

    best_auroc = -1
    best_weights = None

    for epoch in range(EPOCHS):
        model.train()
        for input_ids, attention_mask, y in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=y.to(device),
            )
            out.loss.backward()
            optimizer.step()
            scheduler.step()

        auroc = evaluate(model, val_data, max_len=max_len)["auroc"]
        print(f"stage {stage} lr {lr} epoch {epoch + 1}: val auroc {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model