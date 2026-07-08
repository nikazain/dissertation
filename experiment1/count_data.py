from datasets import load_dataset
import pandas as pd

ds = load_dataset("yaful/MAGE")
train = ds["train"].to_pandas()
test = ds["test"].to_pandas()
allrows = pd.concat([train, test], ignore_index=True)

models = {"A": "text-davinci-002", "B": "text-davinci-003", "C": "gpt-3.5-trubo"}

for stage, tag in models.items():
    n = (allrows["src"].str.endswith(f"continuation_{tag}")).sum()
    print(f"stage {stage} ({tag}): {n} machine continuation rows")

human = (allrows["src"].str.endswith("_human")).sum()
print(f"shared human pool: {human} rows")