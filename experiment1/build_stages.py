import os
import pandas as pd
from datasets import load_dataset

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = {
    "A": "text-davinci-002",
    "B": "text-davinci-003",
    "C": "gpt-3.5-trubo", # typo, be careful
}

TEST_PER_STAGE = 400   # machine rows held out for each frozen test set
HUMAN_TEST = 400       # human rows held out (shared across stages)
HUMAN_TRAIN  = 7800
SEED = 42

# load both splits
print("loading MAGE...")
ds = load_dataset("yaful/MAGE")
train = ds["train"].to_pandas()
test = ds["test"].to_pandas()

# pull continuation rows for one model, from both splits
def machine_rows(model_tag):
    tag = f"continuation_{model_tag}"
    rows = []
    for df in (train, test):
        rows.append(df[df["src"].str.endswith(tag)])
    return pd.concat(rows, ignore_index=True)

# all human continuation-domain rows
def human_rows():
    return train[train["src"].str.endswith("_human")]

#build the shared human pool, split once into test + train 
humans = human_rows().sample(frac=1, random_state=SEED) 
human_test  = humans.iloc[:HUMAN_TEST]
human_train = humans.iloc[HUMAN_TEST:].head(HUMAN_TRAIN)
print(f"human pool: {len(humans)} -> {len(human_train)} train / {len(human_test)} test (shared)")

# build each stage
for stage, model_tag in STAGES.items():
    m = machine_rows(model_tag).sample(frac=1, random_state=SEED)
    m_test = m.iloc[:TEST_PER_STAGE]
    m_train = m.iloc[TEST_PER_STAGE:]

    # train = this stage's machine + shared human
    stage_train = pd.concat([m_train, human_train], ignore_index=True)
    # test = this stage's machine + shared human
    stage_test = pd.concat([m_test, human_test], ignore_index=True)

    stage_train.to_csv(os.path.join(OUT_DIR, f"stage_{stage}_train.csv"), index=False)
    stage_test.to_csv(os.path.join(OUT_DIR, f"stage_{stage}_test.csv"), index=False)

    print(f"stage {stage} ({model_tag}): "
          f"{len(m_train)} machine train, {len(m_test)} machine test")

print("done — 6 files written to", OUT_DIR)