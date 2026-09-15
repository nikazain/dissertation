"""
Generates all figures and tables.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join("figures", "ch3")
DATA = "data"
SAMPLE_PER_CLASS = 25000
SEED = 0

GENERATORS = [
    ("GPT-J-6B",         "2021-06", 1, "EleutherAI",  7588),
    ("T0-3B",            "2021-10", 1, "BigScience",  9452),
    ("T0-11B",           "2021-10", 1, "BigScience",  8930),
    ("GPT-NeoX-20B",     "2022-02", 1, "EleutherAI",  6839),
    ("text-davinci-002", "2022-03", 1, "OpenAI",     21842),
    ("OPT-125M",         "2022-05", 2, "OPT",         8849),
    ("OPT-350M",         "2022-05", 2, "OPT",         8775),
    ("OPT-1.3B",         "2022-05", 2, "OPT",         9202),
    ("OPT-2.7B",         "2022-05", 2, "OPT",         9161),
    ("OPT-6.7B",         "2022-05", 2, "OPT",         8858),
    ("OPT-13B",          "2022-05", 2, "OPT",         8126),
    ("OPT-30B",          "2022-05", 2, "OPT",         8971),
    ("BLOOM-7B",         "2022-07", 3, "BigScience",  8818),
    ("GLM-130B",         "2022-08", 3, "GLM",         9182),
    ("FLAN-T5-small",    "2022-10", 3, "FLAN-T5",     9382),
    ("FLAN-T5-base",     "2022-10", 3, "FLAN-T5",     9454),
    ("FLAN-T5-large",    "2022-10", 3, "FLAN-T5",     9394),
    ("FLAN-T5-XL",       "2022-10", 3, "FLAN-T5",     9170),
    ("FLAN-T5-XXL",      "2022-10", 3, "FLAN-T5",     9319),
    ("text-davinci-003", "2022-11", 4, "OpenAI",     22092),
    ("OPT-IML-1.3B",     "2022-12", 4, "OPT",         9330),
    ("OPT-IML-30B",      "2022-12", 4, "OPT",         9157),
    ("LLaMA-7B",         "2023-02", 5, "LLaMA",       9278),
    ("LLaMA-13B",        "2023-02", 5, "LLaMA",       9290),
    ("LLaMA-30B",        "2023-02", 5, "LLaMA",       9351),
    ("LLaMA-65B",        "2023-02", 5, "LLaMA",       9335),
    ("GPT-3.5-turbo",    "2023-03", 5, "OpenAI",     22679),
]

NAVY, CORAL, TEAL = "#2C4F63", "#EE6A56", "#4EC0B9"
AMBER, PLUM, STEEL, LAVENDER = "#F0A63C", "#8E7BA8", "#7C9EB6", "#D8D5E6"
PALETTE = [NAVY, CORAL, TEAL, AMBER, PLUM, STEEL, LAVENDER]

FAM_COL = {"OpenAI": NAVY, "OPT": CORAL, "FLAN-T5": TEAL,
           "BigScience": AMBER, "LLaMA": PLUM,
           "EleutherAI": STEEL, "GLM": LAVENDER}
INK = "#1A1A1A"

plt.rcParams.update({
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.linewidth": 1.0, "axes.edgecolor": "#444444",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC", "legend.framealpha": 1.0,
    "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def ink_for(colour):
    """White text on dark fills, ink on light ones."""
    r, g, b = mcolors.to_rgb(colour)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


def house_line(ax, x, y, colour, label):
    """The line style of the reference chart: solid line, white-faced dots."""
    ax.plot(x, y, color=colour, linewidth=2.2, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgecolor=colour,
            markeredgewidth=1.6, label=label, zorder=3,
            solid_capstyle="round")


def write_booktabs(path, header, rows, bold_max_cols=None):
    """Plain academic LaTeX table in the reference style (booktabs)."""
    lines = ["\\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for row in rows:
        cells = [str(v) for v in row]
        if bold_max_cols:
            for c in bold_max_cols:
                cells[c] = "\\textbf{" + cells[c] + "}"
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
STAGE_TOTALS = {s: sum(n for _, _, st, _, n in GENERATORS if st == s)
                for s in range(1, 6)}

SHORT = {"text-davinci-002": "davinci-002", "text-davinci-003": "davinci-003",
         "OPT-125M": "125M", "OPT-350M": "350M", "OPT-1.3B": "1.3B",
         "OPT-2.7B": "2.7B", "OPT-6.7B": "6.7B", "OPT-13B": "13B",
         "OPT-30B": "30B", "OPT-IML-1.3B": "IML-1.3B",
         "OPT-IML-30B": "IML-30B", "FLAN-T5-small": "small",
         "FLAN-T5-base": "base", "FLAN-T5-large": "large",
         "FLAN-T5-XL": "XL", "FLAN-T5-XXL": "XXL", "LLaMA-7B": "7B",
         "LLaMA-13B": "13B", "LLaMA-30B": "30B", "LLaMA-65B": "65B"}

def family_legend(ax, y=1.01):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COL.values()]
    ax.legend(handles, FAM_COL.keys(), loc="lower center",
              bbox_to_anchor=(0.5, y), ncol=7, columnspacing=1.1,
              handlelength=1.2)

def figure_lengths():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    lengths = {}
    for label, name in [(0, "machine"), (1, "human")]:
        path = os.path.join(DATA,
                            "machine.parquet" if label == 0 else "human.parquet")
        texts = pd.read_parquet(path, columns=["text"])["text"]
        texts = texts.sample(min(SAMPLE_PER_CLASS, len(texts)),
                             random_state=SEED).tolist()
        out = []
        for i in range(0, len(texts), 2000):
            enc = tok(texts[i:i + 2000], truncation=False)["input_ids"]
            out.extend(len(x) for x in enc)
        lengths[name] = np.array(out)
        print(f"{name}: n={len(out)}  p95={np.percentile(out, 95):.0f}")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.logspace(np.log10(10), np.log10(4000), 60)
    for name, colour in [("machine", CORAL), ("human", NAVY)]:
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
    fig.savefig(os.path.join(OUT, "lengths.pdf"))
    plt.close(fig)

def figure_composition():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for stage in range(1, 6):
        left = 0
        for name, _, s, fam, n in GENERATORS:
            if s != stage:
                continue
            ax.barh(stage, n, left=left, color=FAM_COL[fam],
                    edgecolor="white", linewidth=0.6, height=0.62)
            if n > 5200:
                ax.text(left + n / 2, stage, SHORT.get(name, name),
                        ha="center", va="center", fontsize=6.6,
                        color=ink_for(FAM_COL[fam]), fontweight="bold")
            left += n
        ax.text(left + 700, stage, f"{left:,}", va="center", fontsize=8)
    family_legend(ax)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f"$\\mathbf{{S_{s}}}$" for s in range(1, 6)])
    ax.invert_yaxis()
    ax.set_xlabel("texts")
    ax.set_xlim(0, 69000)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stage_composition.pdf"), bbox_inches="tight")
    plt.close(fig)

def table_stage_image():
    cols = {s: [(n, d, f) for n, d, s2, f, _ in GENERATORS if s2 == s]
            for s in range(1, 6)}
    nrows = max(len(v) for v in cols.values())
    cell, colours = [], []
    for r in range(nrows):
        row, crow = [], []
        for s in range(1, 6):
            if r < len(cols[s]):
                n, d, fam = cols[s][r]
                row.append(f"{n}\n{pd.Timestamp(d).strftime('%b %Y')}")
                crow.append(mcolors.to_rgb(FAM_COL[fam]))
            else:
                row.append("")
                crow.append((1, 1, 1))
        cell.append(row)
        colours.append(crow)

    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.axis("off")
    labels = [f"$\\mathbf{{S_{s}}}$  ({STAGE_TOTALS[s]:,})"
              for s in range(1, 6)]
    t = ax.table(cellText=cell, colLabels=labels, cellColours=colours,
                 colColours=[(1, 1, 1, 1)] * 5, loc="center",
                 cellLoc="center", colLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.6)
    t.scale(1, 2.35)
    for (r, c), cl in t.get_celld().items():
        cl.set_edgecolor("white")
        cl.set_linewidth(1.6)
        fill = cl.get_facecolor()
        cl.set_text_props(color=ink_for(fill),
                          fontweight="bold" if r == 0 else "normal")
        if r == 0:
            cl.set_text_props(color=INK, fontweight="bold")
            cl.set_fontsize(8.5)
    family_legend(ax, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table_stages.pdf"), bbox_inches="tight")
    plt.close(fig)

OUT4 = os.path.join("figures", "ch4")

EXP1_MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
EXP1_AUROC = [
    [0.510, 0.430, 0.480, 0.540, 0.420],
    [0.979, 0.972, 0.929, 0.968, 0.969],
    [0.934, 0.997, 0.880, 0.941, 0.819],
    [0.969, 0.988, 0.981, 0.942, 0.939],
    [0.973, 0.991, 0.934, 0.984, 0.966],
    [0.969, 0.984, 0.930, 0.978, 0.981],
]

CUMUL_MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
CUMUL_AUROC = [
    [0.507, 0.433, 0.476, 0.541, 0.420],
    [0.985, 0.977, 0.941, 0.978, 0.979],
    [0.989, 0.998, 0.952, 0.989, 0.959],
    [0.986, 0.987, 0.986, 0.977, 0.976],
    [0.987, 0.995, 0.970, 0.991, 0.976],
    [0.971, 0.963, 0.939, 0.971, 0.990],
]

TRUNC_BUDGETS = [32, 64, 128, 256, 512, 1024]
TRUNC_MODELS = ["M1", "M2", "M3", "M4", "M5", "pooled"]
def _trunc_union(path="results_truncation.json"):
    import json
    tr = json.load(open(path))
    mach = {"S1": 5437, "S2": 6182, "S3": 6447, "S4": 4038, "S5": 5974}
    n = sum(v + 5616 for v in mach.values())
    return [[sum(tr[str(b)][m][s]["accuracy"] * (mach[s] + 5616) for s in mach) / n
             for m in TRUNC_MODELS] for b in TRUNC_BUDGETS]

TRUNC_ACC = _trunc_union()

def _matrix_tex(models, matrix, path):
    """Booktabs table for an M x S matrix. Matched-stage cells in bold."""
    lines = ["\\begin{tabular}{l" + "c" * 5 + "}", "\\toprule",
             " & " + " & ".join(f"$S_{s}$" for s in range(1, 6)) + " \\\\",
             "\\midrule"]
    for r, name in enumerate(models):
        cells = []
        for c in range(5):
            v = f"{matrix[r][c]:.3f}"
            if r == c + 1:                   
                v = "\\textbf{" + v + "}"
            cells.append(v)
        label = f"$M_{name[1:]}$"
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
        if r == 0:                          
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def table_exp1_tex():
    _matrix_tex(EXP1_MODELS, EXP1_AUROC,
                os.path.join(OUT4, "table_exp1_auroc.tex"))

def table_cumulative_tex():
    _matrix_tex(CUMUL_MODELS, CUMUL_AUROC,
                os.path.join(OUT4, "table_cumulative_auroc.tex"))

def table_truncation_tex():
    lines = ["\\begin{tabular}{r" + "c" * len(TRUNC_MODELS) + "}",
             "\\toprule",
             "Budget & " + " & ".join(
                 f"$M_{m[1:]}$" if m.startswith("M") else m
                 for m in TRUNC_MODELS) + " \\\\",
             "\\midrule"]
    for r, budget in enumerate(TRUNC_BUDGETS):
        cells = [f"{v:.2f}" for v in TRUNC_ACC[r]]
        lines.append(f"{budget} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT4, "table_truncation_acc.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

TRUNC_LINE_COL = {"M1": NAVY, "M2": CORAL, "M3": TEAL,
                  "M4": AMBER, "M5": PLUM, "pooled": STEEL}

def figure_truncation_curve():
    acc = np.array(TRUNC_ACC)          
    x = np.arange(len(TRUNC_BUDGETS)) 

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, name in enumerate(TRUNC_MODELS):
        label = f"$M_{name[1:]}$" if name.startswith("M") else name
        house_line(ax, x, acc[:, i], TRUNC_LINE_COL[name], label)

    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in TRUNC_BUDGETS])
    ax.set_xlabel("input token budget")
    ax.set_ylabel("accuracy")
    ax.grid(axis="y", color="#E9E9E9", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=2, columnspacing=1.3,
              handlelength=1.6, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT4, "truncation_curve.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT4, exist_ok=True)
    figure_composition()
    print("composition written")
    table_stage_image()
    print("stage table written")
    figure_lengths()
    print("lengths written")
    table_exp1_tex()
    table_cumulative_tex()
    table_truncation_tex()
    print("ch4 tables written (tex)")
    figure_truncation_curve()
    print("truncation curve written")

if __name__ == "__main__":
    main()