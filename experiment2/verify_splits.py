"""
Checks the frozen parquet files match the methodology
"""

import data_loading as dl
from config import STAGES

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

# The same human rows are used at every stage
fixed_humans = set(dl.human_sample("test")["text"])
for stage in STAGES:
    stage_humans = set(dl.stage_eval(stage, "test").query("label == 1")["text"])
    assert stage_humans == fixed_humans, f"human sample differs at stage {stage}"

# Every training set stays close to balanced
for stage in STAGES:
    share = (dl.stage_train(stage)["label"] == 0).mean()
    assert 0.4 < share < 0.6, (stage, share)

print("all checks passed")