"""
Building datasets from the frozen files produced in prepare_data.py
same human rows for each stage (first N by shuffle rank)
machine rows are also cut to the first N by shuffle_rank when TRAIN_CAP is set

human_sample(split) - the fixed human rows for a split
stage_train(stage) - stage's machine train rows + fixed human sample
stage_eval(stage, split) - stage's machine rows + fixed human sample
pooled_train() - all machine train rows + fixed human sample (the joint upper bound)
tokenize(texts) - turn texts into Longformer inputs (MAX_LEN 1024)
"""

import pandas as pd
from transformers import AutoTokenizer
from config import HUMAN_SIZES, TRAIN_CAP
 
MODEL_NAME = "allenai/longformer-base-4096"
MAX_LEN = 1024
 
MACHINE = pd.read_parquet("data/machine.parquet")
HUMAN = pd.read_parquet("data/human.parquet")
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
 
def tokenize(texts):
    return tokenizer(list(texts),max_length=MAX_LEN,padding="max_length",truncation=True,return_tensors="pt",)
 
def human_sample(split, cap=None):
    n = HUMAN_SIZES[split]
    if cap is not None:
        n = min(n, cap)
    rows = HUMAN[(HUMAN["mage_split"] == split) & (HUMAN["shuffle_rank"] < n)]
    return rows[["text", "label"]]
 
def stage_train(stage):
    machine = MACHINE[(MACHINE["stage"] == stage) & (MACHINE["mage_split"] == "train")]
    if TRAIN_CAP is not None:
        machine = machine[machine["shuffle_rank"] < TRAIN_CAP]
    data = pd.concat([machine[["text", "label"]], human_sample("train", TRAIN_CAP)])
    return data.sample(frac=1, random_state=stage).reset_index(drop=True)
 
def stage_eval(stage, split):
    machine = MACHINE[(MACHINE["stage"] == stage) & (MACHINE["mage_split"] == split)]
    data = pd.concat([machine[["text", "label"]], human_sample(split)])
    return data.reset_index(drop=True)
 
def pooled_train():
    machine = MACHINE[MACHINE["mage_split"] == "train"]
    if TRAIN_CAP is not None:
        # ranks were assigned within (split, stage), so this caps every stage
        machine = machine[machine["shuffle_rank"] < TRAIN_CAP]
    data = pd.concat([machine[["text", "label"]], human_sample("train", TRAIN_CAP)])
    return data.sample(frac=1, random_state=0).reset_index(drop=True)