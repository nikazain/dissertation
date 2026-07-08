from train import new_model, train_stage

model = train_stage(new_model(), stage=1, lr=3e-5, limit=200, max_len=256, batch_size=4)
print("smoke test finished")