"""
Generates the Data chapter figures and tables.

Figure 3.1  token length histogram per class, log x axis, lines at 512/1024
Figure 3.2  the 27 generators on a release-date axis, one row each, by stage
Table (domains)     figures/table_domains.tex     LaTeX, ready to \input
Table (generators)  figures/table_generators.tex  LaTeX, ready to \input

Run from inside experiment3/:
    python make_data_figures.py
Reads ../experiment2/data/*.parquet. Outputs into figures/.

Release dates are month-precision and hardcoded below. Verify them against
the models' public announcements before submission.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

E2 = os.path.join("..", "experiment2")
OUT = "figures"
SAMPLE_PER_CLASS = 25000
SEED = 0

# (display name, release year-month, stage)
GENERATORS = [
    ("GPT-J-6B",          "2021-06", 1),
    ("T0-3B",             "2021-10", 1),
    ("T0-11B",            "2021-10", 1),
    ("GPT-NeoX-20B",      "2022-02", 1),
    ("text-davinci-002",  "2022-03", 1),
    ("OPT-125M",          "2022-05", 2),
    ("OPT-350M",          "2022-05", 2),
    ("OPT-1.3B",          "2022-05", 2),
    ("OPT-2.7B",          "2022-05", 2),
    ("OPT-6.7B",          "2022-05", 2),
    ("OPT-13B",           "2022-05", 2),
    ("OPT-30B",           "2022-05", 2),
    ("BLOOM-7B",          "2022-07", 3),
    ("GLM-130B",          "2022-08", 3),
    ("FLAN-T5-small",     "2022-10", 3),
    ("FLAN-T5-base",      "2022-10", 3),
    ("FLAN-T5-large",     "2022-10", 3),
    ("FLAN-T5-XL",        "2022-10", 3),
    ("FLAN-T5-XXL",       "2022-10", 3),
    ("text-davinci-003",  "2022-11", 4),
    ("OPT-IML-1.3B",      "2022-12", 4),
    ("OPT-IML-30B",       "2022-12", 4),
    ("LLaMA-7B",          "2023-02", 5),
    ("LLaMA-13B",         "2023-02", 5),
    ("LLaMA-30B",         "2023-02", 5),
    ("LLaMA-65B",         "2023-02", 5),
    ("GPT-3.5-turbo",     "2023-03", 5),
]
STAGE_COLOURS = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868",
                 4: "#C44E52", 5: "#8172B3"}

# (domain, description) for the domains table
DOMAINS = [
    ("CMV",       "opinion and argument posts (Reddit ChangeMyView)"),
    ("Yelp",      "business and restaurant reviews"),
    ("XSum",      "one-sentence news summaries (BBC)"),
    ("TLDR",      "short summaries of Reddit posts"),
    ("ELI5",      "plain-language explanations (Reddit ELI5)"),
    ("WP",        "creative stories (Reddit WritingPrompts)"),
    ("ROC",       "short commonsense stories (ROCStories)"),
    ("HellaSwag", "commonsense sentence continuations"),
    ("SQuAD",     "Wikipedia-based question answering"),
    ("SciXGen",   "scientific paper text"),
]


def token_lengths():
    tok = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    out = {}
    for label, name in [(0, "machine"), (1, "human")]:
        path = os.path.join(E2, "data",
                            "machine.parquet" if label == 0 else "human.parquet")
        texts = pd.read_parquet(path, columns=["text"])["text"]
        texts = texts.sample(min(SAMPLE_PER_CLASS, len(texts)),
                             random_state=SEED).tolist()
        lengths = []
        for i in range(0, len(texts), 2000):
            enc = tok(texts[i:i + 2000], truncation=False)["input_ids"]
            lengths.extend(len(x) for x in enc)
        out[name] = np.array(lengths)
        print(f"{name}: n={len(lengths)}  p95={np.percentile(lengths, 95):.0f}")
    return out


def figure_lengths(lengths):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.logspace(np.log10(10), np.log10(4000), 60)
    for name, colour in [("machine", "#C44E52"), ("human", "#4C72B0")]:
        ax.hist(lengths[name], bins=bins, alpha=0.55, label=name,
                color=colour, edgecolor="none")
    for x, style in [(512, "--"), (1024, "-")]:
        ax.axvline(x, color="black", linestyle=style, linewidth=1)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {x}", va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("tokens (log scale)")
    ax.set_ylabel("texts")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig_3_1_lengths.{ext}"), dpi=200)
    plt.close(fig)


def figure_timeline():
    """One row per generator inside its stage band, so labels never collide."""
    fig, ax = plt.subplots(figsize=(9, 6))
    by_stage = {}
    for name, d, stage in GENERATORS:
        by_stage.setdefault(stage, []).append((pd.Timestamp(d), name))
    for stage, items in by_stage.items():
        items.sort()
        n = len(items)
        for i, (date, name) in enumerate(items):
            y = stage - 0.32 + i * (0.64 / max(n - 1, 1))
            ax.scatter(date, y, color=STAGE_COLOURS[stage], s=22, zorder=3)
            ax.annotate(name, (date, y), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, va="center")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([f"stage {s}" for s in range(1, 6)])
    ax.set_ylim(0.4, 5.6)
    ax.margins(x=0.14)
    ax.set_xlabel("public release date")
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"fig_3_2_timeline.{ext}"), dpi=200)
    plt.close(fig)


def month_name(ym):
    return pd.Timestamp(ym).strftime("%b %Y")


def write_tables():
    # domains table
    lines = ["\\begin{tabular}{ll}", "\\toprule",
             "Domain & Source material \\\\", "\\midrule"]
    for dom, desc in DOMAINS:
        lines.append(f"{dom} & {desc} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_domains.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # generators table, grouped by stage
    lines = ["\\begin{tabular}{lll}", "\\toprule",
             "Stage & Generator & Release \\\\", "\\midrule"]
    last_stage = None
    for name, d, stage in GENERATORS:
        s = str(stage) if stage != last_stage else ""
        if stage != last_stage and last_stage is not None:
            lines.append("\\midrule")
        lines.append(f"{s} & {name} & {month_name(d)} \\\\")
        last_stage = stage
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_generators.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("tables written")


def main():
    os.makedirs(OUT, exist_ok=True)
    write_tables()
    figure_timeline()
    print("figure 3.2 written")
    figure_lengths(token_lengths())
    print("figure 3.1 written")


if __name__ == "__main__":
    main()