"""
Checks the frozen parquet files match the methodology
"""

import numpy as np
import data_loading as dl
from config import STAGES, TRAIN_CAP
 
MACHINE = dl.MACHINE
HUMAN = dl.HUMAN
 
# Totals (in-distribution, after dropping the 4 OOD domains)
assert len(MACHINE) == 281824, len(MACHINE)
assert len(HUMAN) == 150858, len(HUMAN)
 
# Expected per stage machine counts (27 generators across 5 stages)
expected_stage_counts = {1: 54651, 2: 61942, 3: 64719, 4: 40579, 5: 59933}
for stage in STAGES:
    n = (MACHINE["stage"] == stage).sum()
    assert n == expected_stage_counts[stage], (stage, n)
 
# Labels: machine is 0, human is 1
assert (MACHINE["label"] == 0).all()
assert (HUMAN["label"] == 1).all()
 
# Machine shuffle_rank is a clean permutation within every (split, stage)
for split in ["train", "validation", "test"]:
    for stage in STAGES:
        r = MACHINE[
            (MACHINE["mage_split"] == split) & (MACHINE["stage"] == stage)
        ]["shuffle_rank"].to_numpy()
        assert np.array_equal(np.sort(r), np.arange(len(r))), (split, stage)
 
# The same human rows are used at every stage
fixed_humans = set(dl.human_sample("test")["text"])
for stage in STAGES:
    stage_humans = set(dl.stage_eval(stage, "test").query("label == 1")["text"])
    assert stage_humans == fixed_humans, f"human sample differs at stage {stage}"
 
# Every training set stays close to balanced
for stage in STAGES:
    share = (dl.stage_train(stage)["label"] == 0).mean()
    assert 0.4 < share < 0.6, (stage, share)
 
# The cap is respected on both classes (training only)
if TRAIN_CAP is not None:
    for stage in STAGES:
        counts = dl.stage_train(stage)["label"].value_counts()
        assert counts.get(0, 0) <= TRAIN_CAP, (stage, "machine over cap")
        assert counts.get(1, 0) <= TRAIN_CAP, (stage, "human over cap")
 
print("all checks passed")