# HRT — Hinge Regression Tree

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![ICLR 2026](https://img.shields.io/badge/ICLR%202026-Accepted-brightgreen)

<img width="1250" height="381" alt="HRT_github drawio" src="https://github.com/user-attachments/assets/960330f3-685d-45f8-ae1a-e36ff83e5d68" />
<br><br>


**Official implementation** of **Hinge Regression Tree (HRT)** — an optimization-grounded **oblique, piecewise-linear tree** that learns each split by solving a **hinge (max/min) nonlinear least-squares** problem over two linear predictors, giving **ReLU-like expressivity** for **regression** and a **simple binary-classification extension** (fit on \{0,1\} targets; clip to [0,1] and threshold at 0.5).  
(See the ICLR 2026 paper / OpenReview for details.)

> **Companion project:** [HRT-Boost](https://github.com/Hongyi-Li-sz/HRT-Boost) extends HRT from a single oblique piecewise-linear tree to a boosted ensemble of HRT base learners.  
> Use this repository for the original single-tree HRT reference implementation, and use **HRT-Boost** for the boosted compact tabular regression model.


## Paper

**Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting**  
Hongyi Li, Han Lin, Jun Xu*  
**ICLR 2026 (accepted)** 
[PFD ](https://openreview.net/pdf?id=3pZwJB6SX9)


## Relation to HRT-Boost

This repository provides the original single-tree **Hinge Regression Tree (HRT)** reference implementation for the ICLR 2026 paper.

[**HRT-Boost**](https://github.com/Hongyi-Li-sz/HRT-Boost) is the companion extension that uses HRT as the base learner in a boosting framework. It is intended for users who want a compact boosted model for tabular regression.

Use this repository if you want to:

- study or reproduce the original single-tree HRT algorithm;
- inspect the Newton / Gauss-Newton hinge-splitting formulation;
- compare HRT against single-tree baselines.

Use [HRT-Boost](https://github.com/Hongyi-Li-sz/HRT-Boost) if you want to:

- train boosted ensembles of HRT base learners;
- reproduce the HRT-Boost benchmark pipeline;
- compare HRT-Boost with classical and deep tabular baselines.


## Install

Clone the official HRT repository:

```bash
git clone https://github.com/Hongyi-Li-sz/Hinge-Regression-Tree.git
```
```bash
cd Hinge-Regression-Tree/HRT
```

Core (HRT only):

```bash
pip install -e .
```

Run paper baselines (optional extras):

```bash
pip install -e ".[xgb,m5,lt]"
```

## Data

This repo does not bundle datasets. For `kin8nm`, place `kin8nm.data` in the repo root (or pass `--data`).

## Reproduce

This script reproduces the main kin8nm result in the paper.
```bash
python scripts/run_kin8nm.py --data kin8nm.data --seed 42 --reps 5 --pdf results_kin.pdf
```

## Included baselines

- Ridge
- CART
- XGBoost (optional, via `.[xgb]`)
- M5P (optional, via `.[m5]`)
- LinearTree (optional, via `.[lt]`)

## Why HRT?

- **Better splits without heuristics:** oblique split learning is NP-hard and often relies on slow search; HRT instead solves a **nonlinear least-squares problem with hinge (max/min) functions**.
- **Stable optimization:** the node solver aligns with a **damped Newton–Gauss–Newton perspective** and supports **monotone descent** via backtracking.
- **Expressive yet compact:** max/min hinge functions yield **ReLU-like** piecewise-linear models, and empirically HRT is competitive with strong single-tree baselines while often using **fewer leaves**.
- **Second-order optimization formulation:** parameter updates are computed via Newton steps rather than exhaustive search heuristics.



## License

MIT (see `LICENSE`).


## Citation


```bibtex
@inproceedings{li2026hrt,
  title={Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting},
  author={Li, Hongyi and Lin, Han and Xu, Jun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=3pZwJB6SX9}
}
