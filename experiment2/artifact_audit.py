import re
import pandas as pd

CONTROL = [(chr(i), f"control U+{i:04X}") for i in list(range(9, 14)) + [0]]
DETOK = [
    (" .", "space before full stop"),
    (" ,", "space before comma"),
    (" ?", "space before question mark"),
    (" !", "space before exclamation"),
    (" 's", "split possessive ( 's)"),
    (" n't", "split contraction ( n't)"),
    (" 're", "split contraction ( 're)"),
    ("``", "PTB opening quote (``)"),
    ("''", "PTB closing quote ('')"),
    ("( ", "space after open bracket"),
    (" )", "space before close bracket"),
]
MARKUP = [
    ("&amp;", "HTML &amp;"),
    ("&gt;", "HTML &gt;"),
    ("&lt;", "HTML &lt;"),
    ("&quot;", "HTML &quot;"),
    ("&#", "numeric HTML entity"),
    ("<br", "HTML line break"),
    ("[deleted]", "Reddit [deleted]"),
    ("[removed]", "Reddit [removed]"),
    ("http://", "URL http"),
    ("https://", "URL https"),
    ("r/", "subreddit mention"),
]


def check(m, h, items, header):
    print(f"\n {header} (texts containing it)")
    print(f"{'marker':30s} {'machine':>9s} {'human':>9s} {'m rate':>8s} {'h rate':>8s}")
    for pat, name in items:
        mc = int(m.str.contains(pat, regex=False).sum())
        hc = int(h.str.contains(pat, regex=False).sum())
        print(f"{name:30s} {mc:9d} {hc:9d} {mc/len(m):8.4f} {hc/len(h):8.4f}")


def structural(m, h):
    print("\nstructure")
    checks = [
        ("no terminal punctuation", lambda s: ~s.str.rstrip().str[-1:].isin(
            list(".!?\"')"))),
        ("starts with space", lambda s: s.str.startswith(" ")),
        ("ends with space", lambda s: s.str.endswith(" ")),
        ("run of 6+ dots", lambda s: s.str.contains(r"\.{6,}", regex=True)),
        ("run of 6+ exclamations", lambda s: s.str.contains(r"!{6,}", regex=True)),
        ("run of 6+ question marks", lambda s: s.str.contains(r"\?{6,}", regex=True)),
        ("entirely lowercase", lambda s: s == s.str.lower()),
    ]
    print(f"{'check':30s} {'machine':>9s} {'human':>9s}")
    for name, fn in checks:
        print(f"{name:30s} {float(fn(m).mean()):9.4f} {float(fn(h).mean()):9.4f}")


def main():
    m = pd.read_parquet("data/machine.parquet", columns=["text"])["text"]
    h = pd.read_parquet("data/human.parquet", columns=["text"])["text"]
    print(f"machine texts: {len(m):,}   human texts: {len(h):,}")
    check(m, h, CONTROL, "control characters, one by one")
    check(m, h, DETOK, "detokenisation signatures")
    check(m, h, MARKUP, "markup and platform residue")
    structural(m, h)


if __name__ == "__main__":
    main()