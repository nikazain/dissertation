import json, torch
seq = json.load(open("results.json"))
tags = [f"stage{s}_lr{seq['chosen_lr_per_stage'][f'stage{s}']}" for s in range(1, 6)]
prev = torch.load("checkpoints/initial.pt", map_location="cpu") if False else None
sds = [torch.load(f"checkpoints/{t}.pt", map_location="cpu") for t in tags]
from train import new_model
base = new_model().state_dict()                              # M0: pretrained weights + fresh head
chain = [base] + sds
for s in range(1, 6):
    a, b = chain[s - 1], chain[s]
    keys = [k for k in b if k in a and a[k].shape == b[k].shape and b[k].dtype.is_floating_point]
    disp = torch.sqrt(sum(((b[k].float() - a[k].float()) ** 2).sum() for k in keys)).item()
    norm = torch.sqrt(sum((b[k].float() ** 2).sum() for k in keys)).item()
    print(f"M{s-1} -> M{s}  ||dW|| = {disp:9.2f}   relative {disp / norm:.4f}   lr {tags[s-1].split('lr')[1]}")