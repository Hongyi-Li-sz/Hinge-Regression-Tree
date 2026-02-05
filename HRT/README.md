# HRT

Paper: **Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting**  
Authors: Hongyi Li, Han Lin, Jun Xu*  
OpenReview: https://openreview.net/forum?id=3pZwJB6SX9  

Code author/maintainer: Hongyi Li

HRT is a piecewise linear regression tree model (paper-mode release: tune once on a fixed split, then evaluate across repetitions).

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
