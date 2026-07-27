<div align="center">

# diabetes-readmission-dl

**Predicting 30-day hospital readmission with deep learning — and being honest about how well it actually works.**

[Read the paper](reports/paper.md) · [PDF](reports/paper.pdf) · [Notebooks](notebooks/) · [🔬 spin-off tool: leakage-lens](https://github.com/baranlanka/leakage-lens)

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)
![Models](https://img.shields.io/badge/models-DNN%20·%20CNN%20·%20embeddings-8A2BE2)
![Eval](https://img.shields.io/badge/evaluation-patient--grouped%20·%205--seed%20CI-brightgreen)

<img src="reports/figures/fig16_winning_model_roc_pr.png" width="700" alt="ROC and precision-recall curves for the best model on the held-out, patient-grouped test set: ROC-AUC 0.665, PR-AUC 0.220.">

</div>

## Overview

An end-to-end deep-learning study on the **UCI Diabetes 130-US-Hospitals** benchmark (101,766 diabetic inpatient encounters): given one encounter, will the patient be readmitted within 30 days? Two deep architectures — a **Dense DNN** and a **1D-CNN** (plus a diagnosis-**embedding** variant) — are built, tuned, ensembled, and interpreted, with a logistic-regression control carried throughout.

The goal was **a result that can be shown to be sound, not a headline figure.** The published literature on this dataset disagrees with itself — reported ROC-AUC spans **0.48 to 0.974** on identical data — and [the paper](reports/paper.md) shows most of that spread is a measurement artefact. Every number here was produced under a **patient-grouped split** with all resampling and fitted transforms confined to the training partition.

> 🔬 The methodology finding at the heart of this project — that resampling *before* the train/test split inflates AUROC, and by how much — was extracted and generalized into a standalone, installable tool: **[leakage-lens](https://github.com/baranlanka/leakage-lens)**.

## What makes this more than a leaderboard chase

The most useful findings are the **negative** ones, each well-evidenced:

- **Honest performance:** best model ROC-AUC **0.665**, PR-AUC 0.220, minority-F1 0.275 — *above* the published neural-net range (0.58–0.61) and the deployed LACE clinical score (0.56), inside the gradient-boosting band.
- **The threshold beats the tuning.** 60 hyperparameter trials per model moved minority-F1 by **−0.002**; moving one number — the decision threshold, chosen on validation — moved it by **+0.27** (0.007 → 0.275).
- **Deep learning barely beats logistic regression** (PR-AUC +0.0056, as predicted), and the DNN and CNN are **statistically indistinguishable** (paired Wilcoxon).
- **The ensemble gains almost nothing** — and the project tests *why*, rather than shipping it uncritically.
- **37.75%** of patients leak across a naive random split; a patient-grouped split removes it.

## What's inside

| Path | Contents |
|---|---|
| [`notebooks/`](notebooks/) | The full study, rendered with outputs: `01` EDA + preprocessing · `02` model building + the 2×2 validation experiment · `03` hyperparameter tuning (RandomSearch / Bayesian / Hyperband) · `04` evaluation, ensembling, repeated-seed CIs, SHAP |
| [`reports/paper.md`](reports/paper.md) | The written paper (illustrated) |
| `artifacts/` | The trained models (`.keras`, `.joblib`) and results JSON |
| [`reports/figures/`](reports/figures/) | All figures |

## The models (held-out, patient-grouped test set)

| Model | ROC-AUC | PR-AUC | minority-F1 (tuned threshold) |
|---|---|---|---|
| Logistic Regression (control) | 0.655 | 0.214 | 0.28 |
| Dense DNN (Model A) | 0.663 | 0.217 | 0.27 |
| 1D-CNN (Model B) | 0.664 | 0.215 | 0.27 |
| DNN + diagnosis embeddings (A2) | 0.663 | 0.212 | 0.27 |

The ~0.66 ceiling holds across every architecture — evidence the limit is the **data**, not the model.

## Reproduce

```bash
git clone https://github.com/baranlanka/diabetes-readmission-dl && cd diabetes-readmission-dl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_data.py          # downloads the UCI dataset into ./data/
jupyter lab                          # run notebooks 01 → 04 in order
```

Notebook 01 regenerates the cached feature matrices; 02–04 consume them. Seeds are fixed throughout.

## Limitations

Single-machine; neural AUROCs carry ≈±0.005 run-to-run nondeterminism (TensorFlow/oneDNN), which is why headline comparisons use 5-seed confidence intervals. The honest ceiling on this dataset (~0.66) is low by design — the value here is the *evaluation discipline*, not the score.

## Acknowledgments

Data: UCI **Diabetes 130-US-Hospitals** (Strack et al., 2014). Full 29-reference review in [the paper](reports/paper.md).

## License

MIT — see [`LICENSE`](LICENSE).
