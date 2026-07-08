from datasets import load_dataset
import pandas as pd

dataset = load_dataset("yaful/MAGE")

print(dataset) # splits and sizes
test = dataset["test"].to_pandas()

print("\n--- first 5 rows ---")
print(test.head())

print("\n--- label counts (0 = machine, 1 = human) ---")
print(test["label"].value_counts())

print("\n--- a few source strings ---")
print(test["src"].drop_duplicates().head(20))

print("\n--- how many distinct tags in total? ---")
print(test["src"].nunique())

print("\n--- the generator model names ---")
machine = test[test["label"] == 0]
model_names = machine["src"].str.split("_machine_").str[1]
print(model_names.value_counts()) # there are 338 distinct tags, but we should have 340 for all combinations

print("\n--- machine rows whose tag has no '_machine_' in it ---")
machine = test[test["label"] == 0]
weird = machine[~machine["src"].str.contains("_machine_")]
print(weird["src"].value_counts())

print("\n--- find the missing domain/model combos ---")
normal = machine[machine["src"].str.contains("_machine_")]
domains = normal["src"].str.split("_machine_").str[0]
models = normal["src"].str.split("_machine_").str[1]

expected = set()
for d in domains.unique():
    for m in models.unique():
        expected.add(d + "_machine_" + m)

actual = set(normal["src"].unique())
print(sorted(expected - actual))