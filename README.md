# HRT — Hinge Regression Tree

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![ICLR 2026](https://img.shields.io/badge/ICLR%202026-Accepted-brightgreen)

**Official implementation** of **Hinge Regression Tree (HRT)** — an oblique piecewise-linear regression tree that learns each split by solving a **hinge (max/min) nonlinear least-squares problem over two linear predictors**, yielding **ReLU-like expressivity**. The node solver can be interpreted as a **damped Newton / Gauss–Newton** method with stable convergence behavior, and the resulting model class enjoys **universal approximation** with an explicit **O(δ²)** rate under standard conditions.  
(Preprint / camera-ready in preparation.)  
*Maintainer:* Hongyi Li

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

## Reproduce (paper-mode)

```bash
python scripts/run_kin8nm.py --data kin8nm.data --seed 42 --reps 5 --pdf results_kin.pdf
```

## Included baselines

- Ridge
- CART
- XGBoost (optional, via `.[xgb]`)
- M5P (optional, via `.[m5]`)
- LinearTree (optional, via `.[lt]`)

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
