# HRT — Hinge Regression Tree

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![ICLR 2026](https://img.shields.io/badge/ICLR%202026-Accepted-brightgreen)

**Official implementation** of **Hinge Regression Tree (HRT)** — an optimization-grounded **oblique, piecewise-linear tree** that learns each split by solving a **hinge (max/min) nonlinear least-squares** problem over two linear predictors, giving **ReLU-like expressivity** for **regression** and a **simple binary-classification extension** (fit on \{0,1\} targets; clip to [0,1] and threshold at 0.5).  
(See the ICLR 2026 paper / OpenReview for details.)


## Paper

**Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting**  
Hongyi Li, Han Lin, Jun Xu*  
**ICLR 2026 (accepted)** — *camera-ready forthcoming*  
OpenReview: https://openreview.net/forum?id=3pZwJB6SX9  


## Install

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

- **Better splits without heuristics:** oblique split learning is NP-hard and often relies on slow search; HRT instead solves a **hinge (max/min) nonlinear least-squares** split subproblem.
- **Stable optimization:** the node solver aligns with a **damped Newton / Gauss–Newton** view and supports **monotone descent** via backtracking.
- **Expressive yet compact:** max/min hinges yield **ReLU-like** piecewise-linear models, and empirically HRT is competitive with strong single-tree baselines while often using **fewer leaves**.



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
