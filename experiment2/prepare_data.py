"""
Preparing frozen splits 
machine rows (text, label, stage, mage_split)
human rows (text, label, mage_split, shuffle_rank)

`shuffle_rank` creates a fixed random ordering of human texts, so every stage takes N human examples and so uses the same human dataset
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from config import GENERATOR_STAGE, HF_DATASET, SEED

MAGE_DOMAINS = ["cmv", "eli5", "tldr", "xsum", "wp", "roct", "hswag", "yelp", "squad", "sci_gen"]

def find_stage(src): # return continual learning stage for machine generated row
    if src.endswith("_human"):
        return None
    for token, stage in GENERATOR_STAGE.items():
        if src.endswith("_" + token):
            return stage
    raise ValueError(f"Could not identify generator from src: {src}")


def main():
    os.makedirs("data", exist_ok=True)

    # Load all three splits and tag each row with which split it came from
    frames = []
    for split in ["train", "validation", "test"]:
        df = load_dataset(HF_DATASET)[split].to_pandas()
        df["mage_split"] = split
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    starts = tuple(d + "_" for d in MAGE_DOMAINS)
    data = data[data["src"].str.startswith(starts)].reset_index(drop=True)

    machine = data[data["label"] == 0].copy() # split into machine and human rows
    human = data[data["label"] == 1].copy()

    machine["stage"] = machine["src"].apply(find_stage)

    # Seeded per-split shuffle of the human rows -> shuffle_rank
    rng = np.random.default_rng(SEED)
    human["shuffle_rank"] = -1
    for split in ["train", "validation", "test"]:
        idx = human.index[human["mage_split"] == split].to_numpy()
        human.loc[rng.permutation(idx), "shuffle_rank"] = np.arange(len(idx))

    machine = machine[["text", "label", "stage", "mage_split"]]
    human = human[["text", "label", "mage_split", "shuffle_rank"]]

    machine.to_parquet("data/machine.parquet", index=False)
    human.to_parquet("data/human.parquet", index=False)

    print(f"machine rows: {len(machine)}")
    print(f"human rows:   {len(human)}")
    print("stage counts:")
    print(machine["stage"].value_counts().sort_index())


if __name__ == "__main__":
    main()