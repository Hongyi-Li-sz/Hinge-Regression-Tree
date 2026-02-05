from __future__ import annotations

import os
import random
import numpy as np

def set_global_seed(seed: int | None) -> None:
    """Best-effort global seeding (numpy + python random + hash seed)."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
