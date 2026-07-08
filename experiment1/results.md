text = """# Experiment 1 — Sequential fine-tuning, OpenAI generations

Stages: A = davinci-002, B = davinci-003, C = gpt-3.5-turbo
Fresh Longformer for A; each stage continues the previous detector.
1 epoch, max_len 512, batch 4, lr 2e-5, seed 42.

## Full results — every detector on every frozen test set

| Detector     | Test set | HumanRec | MachineRec | AvgRec | AUROC |
| ------------ | -------- | -------- | ---------- | ------ | ----- |
| Detector A   | Test A   | 0.695    | 0.970      | 0.833  | 0.963 |
| Detector A   | Test B   | 0.695    | 0.973      | 0.834  | 0.960 |
| Detector A   | Test C   | 0.695    | 0.995      | 0.845  | 0.981 |
| Detector AB  | Test A   | 0.835    | 0.938      | 0.886  | 0.963 |
| Detector AB  | Test B   | 0.835    | 0.990      | 0.913  | 0.989 |
| Detector AB  | Test C   | 0.835    | 0.988      | 0.911  | 0.993 |
| Detector ABC | Test A   | 0.925    | 0.780      | 0.853  | 0.936 |
| Detector ABC | Test B   | 0.925    | 0.935      | 0.930  | 0.983 |
| Detector ABC | Test C   | 0.925    | 1.000      | 0.963  | 1.000 |

## AvgRec (AUROC) — compact grid

|              | Test A       | Test B       | Test C       |
| ------------ | ------------ | ------------ | ------------ |
| Detector A   | 0.833 (.963) | 0.834 (.960) | 0.845 (.981) |
| Detector AB  | 0.886 (.963) | 0.913 (.989) | 0.911 (.993) |
| Detector ABC | 0.853 (.936) | 0.930 (.983) | 0.963 (1.00) |

## Finding

No catastrophic forgetting. Positive forward transfer (Stage A improved after
training on B: 0.833 -> 0.886). Mild boundary drift at Stage C: detector relaxed
toward human, cutting oldest-generation MachineRec (0.97 -> 0.78) while AUROC
stayed ~0.94. Final detector strong on all three.

Caveats: single run, within-family only.
Next: threshold recalibration (RQ2); cross-family staging (Exp 2).
"""

with open("results.md", "w") as f:
f.write(text)
