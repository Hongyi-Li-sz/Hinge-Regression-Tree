from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np

@dataclass
class Node:
    is_leaf: bool
    region: List[Tuple[float, float]]
    params: Optional[np.ndarray] = None          # (d+1,)
    split_coeffs: Optional[np.ndarray] = None    # (d+1,)
    children: Optional[List["Node"]] = None
    stop_reason: Optional[str] = None
    data_indices: Optional[np.ndarray] = None

    def count_leaves(self) -> int:
        if self.is_leaf:
            return 1
        if not self.children:
            return 0
        return sum(c.count_leaves() for c in self.children)

def linear_model_nd(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    X = np.asanyarray(X)
    X_des = np.hstack([X, np.ones((X.shape[0], 1))])
    return X_des @ coeffs

def predict_nd(X_new: np.ndarray, root_node: Optional[Node]) -> np.ndarray:
    if root_node is None:
        return np.zeros(X_new.shape[0], dtype=float)
    X_des = np.hstack([X_new, np.ones((X_new.shape[0], 1))])

    def _predict_single(x_des_row: np.ndarray, node: Node) -> float:
        if node.is_leaf:
            if node.params is None or np.any(~np.isfinite(node.params)):
                return 0.0
            return float(x_des_row @ node.params)
        assert node.children is not None and node.split_coeffs is not None
        if (x_des_row @ node.split_coeffs) < -1e-9:
            return _predict_single(x_des_row, node.children[0])
        return _predict_single(x_des_row, node.children[1])

    return np.array([_predict_single(r, root_node) for r in X_des], dtype=float)

def collect_leaf_stop_reasons(node: Optional[Node], counts: Dict[str, int]) -> None:
    if node is None:
        return
    if node.is_leaf:
        key = node.stop_reason or "unknown"
        counts[key] = counts.get(key, 0) + 1
        return
    if node.children:
        for c in node.children:
            collect_leaf_stop_reasons(c, counts)
