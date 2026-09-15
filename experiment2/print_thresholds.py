import json
import re

METRIC = "avg_rec" 

with open("results_thresholds.json") as f:
    data = json.load(f)

records = []

def walk(node, path):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)])
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)])
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        records.append((path, float(node)))

walk(data, [])

def role(part):
    p = part.lower()
    if re.fullmatch(r"s[1-5]|stage[1-5]|s_[1-5]|[1-5]", p):
        return "stage"
    if re.fullmatch(r"m[0-9]|pooled|final|m_[0-9]", p):
        return "model"
    if any(w in p for w in ("fixed", "deploy", "refit")):
        return "regime"
    return "metric"

cells = []
for path, value in records:
    r = {"regime": "", "model": "", "stage": "", "metric": ""}
    for part in path:
        r[role(part)] = part
    if r["model"] and r["stage"]:
        cells.append((r["regime"], r["model"], r["stage"], r["metric"], value))

metrics_found = sorted({c[3] for c in cells})
use_metric = METRIC if METRIC in metrics_found else ("" if "" in metrics_found else None)
if not cells or use_metric is None:
    print("could not recognize the structure; metrics found:", metrics_found)
    print(json.dumps(data, indent=2)[:800])
    raise SystemExit

cells = [c for c in cells if c[3] == use_metric]

def stage_key(s):
    return int(re.search(r"[1-5]", s).group())

def model_key(m):
    d = re.search(r"[0-9]", m)
    return (0, int(d.group())) if d else (1, m)

regimes = list(dict.fromkeys(c[0] for c in cells))
models = sorted({c[1] for c in cells}, key=model_key)
stages = sorted({c[2] for c in cells}, key=stage_key)
grid = {(c[0], c[1], c[2]): c[4] for c in cells}

def sep(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

def row(items, widths):
    return "| " + " | ".join(c.ljust(w) for c, w in zip(items, widths)) + " |"

for regime in regimes:
    widths = [max(len("Model"), max(len(m) for m in models))] + [5] * len(stages)
    print(f"\n{(regime or 'table').upper()}  ({use_metric or METRIC})")
    print(sep(widths))
    print(row(["Model"] + [f"S{stage_key(s)}" for s in stages], widths))
    print(sep(widths))
    for m in models:
        vals = [f"{grid[(regime, m, s)]:.3f}" if (regime, m, s) in grid else "-" for s in stages]
        print(row([m] + vals, widths))
    print(sep(widths))