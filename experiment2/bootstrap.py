import json, glob, re
import numpy as np
from sklearn.metrics import roc_auc_score
 
B = 1000
rng = np.random.default_rng(42)
dep = json.load(open("results_thresholds.json"))["regimes"]["deployed"]
out = {}
for f in sorted(glob.glob("probs/trunc1024_*_stage*.npz")):
    m = re.search(r"trunc1024_(.+)_stage(\d)\.npz", f)
    tag, j = m.group(1), int(m.group(2))
    d = np.load(f); p, l = d["probs"], d["labels"]        # labels: 0 = machine, 1 = human
    y = (l == 0).astype(int)                               # machine as positive class
    name = "pooled" if tag.startswith("pooled") else f"M{tag[5]}"
    t = dep.get(name, {}).get(f"S{j}", {}).get("threshold")
    aur, acc = [], []
    for _ in range(B):
        idx = rng.integers(0, len(p), len(p))
        aur.append(roc_auc_score(y[idx], p[idx]))
        if t is not None:
            acc.append(np.mean((p[idx] > t) == y[idx]))
    out[f"{name}/S{j}"] = {"auroc_ci": [float(np.percentile(aur, 2.5)), float(np.percentile(aur, 97.5))],
                           "acc_ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))] if acc else None}
    hw = (out[f"{name}/S{j}"]["auroc_ci"][1] - out[f"{name}/S{j}"]["auroc_ci"][0]) / 2
    print(f"{name:7s} S{j}  AUROC 95% CI half-width {hw:.4f}")
json.dump(out, open("results_ci.json", "w"), indent=2)
print("saved results_ci.json")