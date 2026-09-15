"""
Fixed choices for methodology 
"""
import os
TRAIN_CAP = int(os.environ["TRAIN_CAP"]) if "TRAIN_CAP" in os.environ else None

SEED = 42
HF_DATASET = "yaful/MAGE"
OOD_DOMAINS = ("cnn", "dialogsum", "imdb", "pubmed") # excluded 
GENERATOR_STAGE = {
    # Stage 1
    "gpt_j": 1, "t0_3b": 1, "t0_11b": 1, "gpt_neox": 1, "text-davinci-002": 1,
    # Stage 2
    "opt_125m": 2, "opt_350m": 2, "opt_1.3b": 2, "opt_2.7b": 2,
    "opt_6.7b": 2, "opt_13b": 2, "opt_30b": 2,
    # Stage 3
    "bloom_7b": 3, "GLM130B": 3, "flan_t5_small": 3, "flan_t5_base": 3,
    "flan_t5_large": 3, "flan_t5_xl": 3, "flan_t5_xxl": 3,
    # Stage 4
    "text-davinci-003": 4, "opt_iml_max_1.3b": 4, "opt_iml_30b": 4,
    # Stage 5
    "7B": 5, "13B": 5, "30B": 5, "65B": 5, "gpt-3.5-trubo": 5,
}

STAGES = (1, 2, 3, 4, 5)
HUMAN_SIZES = {"train": 45151, "validation": 5599, "test": 5616} 