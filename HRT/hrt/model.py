from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .solvers import recursive_piecewise_fit_nd, MIN_POINTS_FOR_SPLIT_BASE
from .tree import predict_nd, Node

class HRTRegressor(BaseEstimator, RegressorMixin):
    """HRT: Piecewise linear regression tree model (sklearn-compatible)."""

    def __init__(
        self,
        threshold: float = 0.03,
        min_points: int = MIN_POINTS_FOR_SPLIT_BASE,
        max_depth: int = 5,
        step_size: float | str = 0.1,   # float or "auto"
        ridge_alpha: float = 0.0,
        random_state: int | None = None,
    ):
        self.threshold = threshold
        self.min_points = min_points
        self.max_depth = max_depth
        self.step_size = step_size
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

        self.root_node: Node | None = None
        self.iter_counts_: list[int] = []
        self.n_features_in_: int | None = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self.n_features_in_ = int(X.shape[1])
        self.iter_counts_ = []

        self.root_node = recursive_piecewise_fit_nd(
            X_full=X,
            Z_full=y,
            current_indices=np.arange(len(y)),
            threshold=self.threshold,
            min_points_per_segment=self.min_points,
            depth=0,
            max_depth=self.max_depth,
            current_run_random_state=self.random_state,
            step_size_optimize=self.step_size,
            ridge_alpha=self.ridge_alpha,
            iter_counts_list=self.iter_counts_,
        )
        return self

    def predict(self, X):
        check_is_fitted(self, "root_node")
        X = check_array(X, accept_sparse=False)
        return predict_nd(X, self.root_node)

    def n_leaves(self) -> int:
        check_is_fitted(self, "root_node")
        return 0 if self.root_node is None else self.root_node.count_leaves()
