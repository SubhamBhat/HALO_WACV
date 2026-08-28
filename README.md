# HALO: Homotopic Alignment of Log-Odds for Long-Tailed Visual Recognition

[![Conference](https://img.shields.io/badge/WACV-2027-blue.svg)](https://wacv2027.ieeecomputer.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red.svg)](https://pytorch.org/)

## Method Abstract

Long-tail visual recognition is still dominated by two-stage pipelines built on fixed epoch schedules, linear margin penalties, and manually tuned re-weighting heuristics. We introduce **HALO (Heuristic-Free Adaptive Long-Tail Optimization)**, a continuous framework in which each component is derived from a stated principle in optimization theory, statistical learning theory, and Bayesian decision theory rather than selected by search. We show that naive importance-sampling weights inflate the variance of the weighted gradient estimator on tail classes, and derive a square-root-dampened penalty base that restores an $\mathcal{O}(1/N_c)$ bound. We characterize the gradient shock of polynomial train-accuracy gating and show that the quadratic exponent is the lowest-order schedule whose shock vanishes at both ends of training while peaking at mid-trajectory, unlike linear gating (peak at initialization) or higher-order gating (a near-step transition at convergence). We show that a multiplicative correction of predicted class mass corresponds to an additive logarithmic shift in the log-odds domain, motivating a logarithmic rather than linear penalty. Finally, we derive a Bayes-optimal post-hoc calibration rule with an explicit bound on its temperature search range. Evaluated against eight baselines across five backbones and five benchmarks — coarse-grained (CIFAR-10/100-LT) and fine-grained (Oxford-IIIT Pets-LT, CUB-200-LT, and Stanford Cars-LT, the last with 1–2-shot tail classes) — HALO obtains the best Macro F1 in all 25 backbone–dataset configurations, with significantly narrower margins under extreme tail scarcity, a regime we analyze explicitly. Ablations confirm that each derived term contributes, with quadratic gating and $\ell_1$ trace normalization proving most critical to calibration stability.

## Method Architecture

![HALO Architecture](halo_architecture.jpg)

## Long-Tailed Recognition Evaluation (Macro-F1)

All results are reported as Mean ± Standard Deviation over 3 independent training runs. Bold denotes the best performance.

### ResNet-18

| Method | CIFAR-10-LT | CIFAR-100-LT | Pets-LT | CUB-200-LT | Cars-LT | Mean-F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| WCE | 64.41±0.61 | 36.11±0.12 | 55.88±0.32 | 17.67±0.28 | 16.16±0.69 | 38.05±0.64 |
| CB Loss | 64.94±0.81 | 34.45±0.17 | 56.18±0.44 | 18.06±0.12 | 16.66±0.27 | 38.06±0.50 |
| Focal | 57.97±0.12 | 34.12±0.26 | 50.71±0.62 | 17.82±0.54 | 18.49±0.28 | 35.82±0.57 |
| LDAM | 62.54±0.75 | 32.72±0.11 | 55.53±0.74 | 19.63±0.66 | 10.19±0.37 | 36.12±0.22 |
| DBM | 66.54±0.87 | 35.18±0.37 | 55.98±0.17 | 22.85±0.18 | 16.65±0.78 | 39.44±0.58 |
| ALPA | 59.84±0.75 | 34.60±0.68 | 51.61±0.53 | 18.22±0.88 | 17.67±0.40 | 36.39±0.54 |
| RobustFocal | 61.29±0.76 | 33.26±0.59 | 52.50±0.79 | 15.13±0.56 | 20.80±0.66 | 36.60±0.14 |
| Pub. DCAL | 65.36±0.28 | 35.45±0.33 | 60.20±0.16 | 21.04±0.29 | 15.97±0.18 | 39.60±0.32 |
| **HALO (Ours)** | **73.65±0.61** | **40.58±0.39** | **66.96±0.40** | **25.77±0.27** | **21.73±0.31** | **45.74±0.85** |

### DenseNet-121

| Method | CIFAR-10-LT | CIFAR-100-LT | Pets-LT | CUB-200-LT | Cars-LT | Mean-F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| WCE | 74.62±0.62 | 42.35±0.59 | 64.50±0.24 | 22.59±0.68 | 23.49±0.23 | 45.51±0.40 |
| CB Loss | 74.19±0.89 | 42.97±0.61 | 61.59±0.55 | 24.03±0.65 | 23.36±0.77 | 45.23±0.72 |
| Focal | 72.22±0.28 | 40.68±0.13 | 57.81±0.35 | 24.67±0.31 | 23.31±0.27 | 43.74±0.85 |
| LDAM | 72.21±0.80 | 40.05±0.35 | 65.26±0.62 | 27.25±0.42 | 18.57±0.83 | 44.67±0.47 |
| DBM | 76.27±0.31 | 44.81±0.30 | 64.63±0.55 | 34.73±0.31 | 23.11±0.57 | 48.71±0.82 |
| ALPA | 72.83±0.42 | 41.26±0.28 | 49.21±0.90 | 25.94±0.51 | 28.29±0.17 | 43.51±0.14 |
| RobustFocal | 72.37±0.19 | 40.66±0.60 | 59.48±0.73 | 26.48±0.44 | 27.10±0.15 | 45.22±0.41 |
| Pub. DCAL | 75.38±0.90 | 42.52±0.52 | 63.45±0.88 | 24.72±0.79 | 21.81±0.11 | 45.58±0.68 |
| **HALO (Ours)** | **81.87±0.65** | **48.04±0.53** | **74.02±0.31** | **36.63±0.61** | **28.69±0.19** | **53.85±0.45** |

### SqueezeNet

| Method | CIFAR-10-LT | CIFAR-100-LT | Pets-LT | CUB-200-LT | Cars-LT | Mean-F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| WCE | 64.38±0.46 | 25.94±0.86 | 33.64±0.80 | 11.33±0.31 | 8.26±0.50 | 28.71±0.24 |
| CB Loss | 63.98±0.83 | 26.72±0.80 | 34.47±0.34 | 13.10±0.61 | 5.97±0.59 | 28.85±0.22 |
| Focal | 49.47±0.71 | 22.99±0.53 | 31.08±0.72 | 11.80±0.52 | 8.51±0.10 | 24.77±0.36 |
| LDAM | 42.51±0.12 | 1.85±0.84 | 4.53±0.80 | 0.75±0.77 | 0.44±0.35 | 10.02±0.15 |
| DBM | 40.98±0.80 | 1.50±0.86 | 5.76±0.17 | 0.63±0.49 | 0.44±0.16 | 9.86±0.71 |
| ALPA | 48.61±0.71 | 23.74±0.20 | 29.75±0.48 | 12.85±0.54 | 7.20±0.31 | 24.43±0.80 |
| RobustFocal | 50.01±0.44 | 23.79±0.27 | 31.41±0.53 | 14.13±0.68 | 9.20±0.26 | 25.71±0.35 |
| Pub. DCAL | 62.89±0.90 | 28.48±0.62 | 39.31±0.45 | 16.35±0.51 | 5.71±0.20 | 30.55±0.28 |
| **HALO (Ours)** | **66.27±0.37** | **35.33±0.57** | **50.65±0.28** | **18.89±0.28** | **13.92±0.16** | **37.01±0.60** |



---

## 🚀 Installation & Requirements

`ash
pip install -r requirements.txt
`

## 💻 Training

To train the HALO framework on a specific dataset and backbone:

`ash
# Example: Train HALO on Stanford Cars-LT using ResNet-18
python train.py --dataset stanford_cars --backbone resnet18 --method halo --epochs 100
`

### Reproducing Baseline Comparisons

We provide implementations of several competitive baselines (Published DCAL, LDAM, Focal Loss, etc.) in core/baselines.py.

`ash
python train.py --dataset stanford_cars --backbone resnet18 --method dcal
`
