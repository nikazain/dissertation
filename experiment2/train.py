import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import data_loading as dl
from data_loading import MODEL_NAME, MAX_LEN, tokenizer
from metrics import evaluate
from seeding import get_device

EPOCHS = 5
MICRO_BATCH = 8    # rows per forward/backward pass
ACCUM_STEPS = 8    # optimizer step every 8 micro-batches -> effective 64
WARMUP_RATIO = 0.06
USE_AMP = True     # fp16 on CUDA; ignored elsewhere

def new_model():
    """A fresh Longformer with a 2-class classification head."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def train_stage(model, stage, lr, limit=None, max_len=MAX_LEN,
                batch_size=MICRO_BATCH, accum_steps=ACCUM_STEPS,
                train_df=None, val_df=None):
    device = get_device()
    model.to(device)
    train_data = dl.stage_train(stage) if train_df is None else train_df
    val_data = dl.stage_eval(stage, "validation") if val_df is None else val_df
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
    steps_per_epoch = -(-len(loader) // accum_steps)  # ceil
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps
    )

    amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_auroc = -1
    best_weights = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for i, (input_ids, attention_mask, y) in enumerate(loader):
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=y.to(device),
                )
                loss = out.loss / accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        auroc = evaluate(model, val_data, max_len=max_len)["auroc"]
        print(f"stage {stage} lr {lr} epoch {epoch + 1}: val auroc {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model