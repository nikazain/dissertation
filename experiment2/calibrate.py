import time

from config import SEED, TRAIN_CAP
from seeding import seed_everything
from train import EPOCHS, new_model, train_stage

seed_everything(SEED)
print(f"TRAIN_CAP = {TRAIN_CAP}")

start = time.time()
model = train_stage(new_model(), stage=1, lr=3e-5)
total = time.time() - start

print(f"total: {total / 60:.1f} min")
print(f"per epoch (incl. val pass): {total / EPOCHS / 60:.1f} min")