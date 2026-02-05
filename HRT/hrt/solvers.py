from __future__ import annotations

import numpy as np
from .utils import rmse
from .tree import Node, linear_model_nd

MIN_POINTS_FOR_SPLIT_BASE = 5

def solve_ols_nd(X: np.ndarray, Z: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    n_points, n_features = X.shape
    X_des = np.hstack([X, np.ones((n_points, 1))])
    I = np.eye(n_features + 1)
    I[-1, -1] = 0.0  # do not regularize bias
    if alpha <= 1e-9:
        return np.linalg.solve(X_des.T @ X_des, X_des.T @ Z)
    return np.linalg.solve(X_des.T @ X_des + alpha * I, X_des.T @ Z)

def fit_single_plane(X: np.ndarray, Z: np.ndarray, ridge_alpha: float = 0.0) -> np.ndarray:
    return solve_ols_nd(X, Z, alpha=ridge_alpha)

def initialize_thetas_nd(X: np.ndarray, Z: np.ndarray, random_seed=None, ridge_alpha: float = 0.0):
    n_points, n_features = X.shape
    mean_z = np.mean(Z) if n_points > 0 else 0.0
    min_points_to_fit = n_features + 1
    if n_points < min_points_to_fit:
        base = np.array([0.0] * n_features + [mean_z], dtype=float)
        return base.copy(), base.copy()

    if random_seed is not None:
        np.random.seed(int(random_seed))

    chosen_split_dim = next((dim for dim in range(n_features) if np.max(X[:, dim]) > np.min(X[:, dim])), -1)
    if chosen_split_dim == -1:
        base = np.array([0.0] * n_features + [mean_z], dtype=float)
        return base.copy(), base.copy()

    split_dim = chosen_split_dim
    median_val = np.median(X[:, split_dim])
    mask_left_init = X[:, split_dim] < median_val
    mask_right_init = ~mask_left_init

    if np.sum(mask_left_init) < min_points_to_fit or np.sum(mask_right_init) < min_points_to_fit:
        theta_global = solve_ols_nd(X, Z, alpha=ridge_alpha)
        eps = np.random.rand(n_features + 1) * 0.005
        theta1 = np.nan_to_num(theta_global + eps)
        theta2 = np.nan_to_num(theta_global - eps)
        if np.allclose(theta1, theta2, atol=1e-9):
            if n_features > 0:
                theta1[0] += 0.001
        return theta1, theta2

    theta1 = solve_ols_nd(X[mask_left_init], Z[mask_left_init], alpha=ridge_alpha)
    theta2 = solve_ols_nd(X[mask_right_init], Z[mask_right_init], alpha=ridge_alpha)
    if np.allclose(theta1, theta2, atol=1e-6):
        eps = np.random.rand(n_features + 1) * 0.005
        theta1 = np.nan_to_num(theta1 + eps)
        theta2 = np.nan_to_num(theta2 - eps)
        if np.allclose(theta1, theta2, atol=1e-9):
            if n_features > 0:
                theta1[0] += 0.001
    return theta1, theta2

def _rmse_for_thetas_nd(X_des, Z, theta1, theta2, flip_split_direction,
                        min_points_per_segment, min_points_to_fit):
    split_vec = (theta2 - theta1) if flip_split_direction else (theta1 - theta2)
    left_mask = (X_des @ split_vec) < 0
    n_left = int(np.sum(left_mask))
    n_right = X_des.shape[0] - n_left

    if (n_left < min_points_per_segment or n_right < min_points_per_segment or
        n_left < min_points_to_fit or n_right < min_points_to_fit):
        return False, np.inf, left_mask, ~left_mask

    preds = np.empty_like(Z, dtype=float)
    preds[left_mask] = X_des[left_mask] @ theta1
    preds[~left_mask] = X_des[~left_mask] @ theta2
    return True, rmse(Z, preds), left_mask, ~left_mask

def _run_one_split_direction(
    X, Z, initial_theta1, initial_theta2,
    max_iter, tol, min_points_per_segment, min_points_to_fit,
    step_size, ridge_alpha, flip_split_direction=False
):
    n_points = X.shape[0]
    X_des = np.hstack([X, np.ones((n_points, 1))])
    theta1_current = initial_theta1.copy()
    theta2_current = initial_theta2.copy()
    prev_left_mask = None
    iters = 0

    # auto step backtracking
    step_shrink = 0.5
    step_min = 1e-4
    max_backtracks = 10
    improve_tol = 1e-8

    valid_base, rmse_current, _, _ = _rmse_for_thetas_nd(
        X_des, Z, theta1_current, theta2_current, flip_split_direction,
        min_points_per_segment, min_points_to_fit
    )
    if not valid_base:
        return None, 0

    for i in range(max_iter):
        iters = i + 1
        theta1_old = theta1_current.copy()
        theta2_old = theta2_current.copy()

        prediction_diff = X_des @ ((theta2_current - theta1_current) if flip_split_direction else (theta1_current - theta2_current))
        current_left_mask = prediction_diff < 0
        if prev_left_mask is not None and np.array_equal(current_left_mask, prev_left_mask):
            break
        prev_left_mask = current_left_mask

        n_left = int(np.sum(current_left_mask))
        n_right = n_points - n_left
        if (n_left < min_points_per_segment or n_right < min_points_per_segment or
            n_left < min_points_to_fit or n_right < min_points_to_fit):
            return None, 0

        try:
            theta1_target = solve_ols_nd(X[current_left_mask], Z[current_left_mask], alpha=ridge_alpha)
            theta2_target = solve_ols_nd(X[~current_left_mask], Z[~current_left_mask], alpha=ridge_alpha)
        except np.linalg.LinAlgError:
            theta1_current, theta2_current = theta1_old, theta2_old
            break

        if isinstance(step_size, (int, float)):
            s = float(step_size)
            theta1_try = (1 - s) * theta1_old + s * theta1_target
            theta2_try = (1 - s) * theta2_old + s * theta2_target
            if np.any(~np.isfinite(theta1_try)) or np.any(~np.isfinite(theta2_try)):
                theta1_current, theta2_current = theta1_old, theta2_old
                break
            theta1_current, theta2_current = theta1_try, theta2_try

            if (np.linalg.norm(theta1_current - theta1_old) < tol and
                np.linalg.norm(theta2_current - theta2_old) < tol):
                break

            valid_now, rmse_now, _, _ = _rmse_for_thetas_nd(
                X_des, Z, theta1_current, theta2_current, flip_split_direction,
                min_points_per_segment, min_points_to_fit
            )
            if not valid_now:
                theta1_current, theta2_current = theta1_old, theta2_old
                break
            rmse_current = rmse_now

        elif isinstance(step_size, str) and step_size.lower() == "auto":
            s_try = 1.0
            backtracks = 0
            accepted = False
            while s_try >= step_min and backtracks < max_backtracks:
                theta1_try = (1 - s_try) * theta1_old + s_try * theta1_target
                theta2_try = (1 - s_try) * theta2_old + s_try * theta2_target
                if np.any(~np.isfinite(theta1_try)) or np.any(~np.isfinite(theta2_try)):
                    s_try *= step_shrink
                    backtracks += 1
                    continue
                valid_try, rmse_try, _, _ = _rmse_for_thetas_nd(
                    X_des, Z, theta1_try, theta2_try, flip_split_direction,
                    min_points_per_segment, min_points_to_fit
                )
                if valid_try and (rmse_try < rmse_current - improve_tol):
                    theta1_current, theta2_current = theta1_try, theta2_try
                    rmse_current = rmse_try
                    accepted = True
                    break
                s_try *= step_shrink
                backtracks += 1
            if not accepted:
                theta1_current, theta2_current = theta1_old, theta2_old
                break

            if (np.linalg.norm(theta1_current - theta1_old) < tol and
                np.linalg.norm(theta2_current - theta2_old) < tol):
                break
        else:
            raise ValueError("Unsupported step_size. Use float or 'auto'.")

    final_split_coeffs = (theta2_current - theta1_current) if flip_split_direction else (theta1_current - theta2_current)
    final_left_mask = (X_des @ final_split_coeffs) < 0

    if (np.sum(final_left_mask) < min_points_per_segment or np.sum(~final_left_mask) < min_points_per_segment or
        np.sum(final_left_mask) < min_points_to_fit or np.sum(~final_left_mask) < min_points_to_fit):
        return None, iters

    preds = np.zeros_like(Z, dtype=float)
    preds[final_left_mask] = linear_model_nd(X[final_left_mask], theta1_current)
    preds[~final_left_mask] = linear_model_nd(X[~final_left_mask], theta2_current)
    return (final_split_coeffs, theta1_current, theta2_current, final_left_mask, ~final_left_mask, rmse(Z, preds)), iters

def optimize_split_optimized_nd(
    X, Z, max_iter=100, tol=1e-6,
    min_points_per_segment=MIN_POINTS_FOR_SPLIT_BASE,
    step_size=1, random_seed_for_init=None, ridge_alpha=1e-2
):
    n_points, n_features = X.shape
    min_points_to_fit = n_features + 1
    if n_points < 2 * min_points_to_fit or n_points < 2 * min_points_per_segment:
        return None, None, None, None, None, 0

    theta1_0, theta2_0 = initialize_thetas_nd(X, Z, random_seed=random_seed_for_init, ridge_alpha=ridge_alpha)
    if np.any(~np.isfinite(theta1_0)) or np.any(~np.isfinite(theta2_0)):
        return None, None, None, None, None, 0

    best_res = None
    best_rmse = float("inf")
    best_iters = 0

    res1, iters1 = _run_one_split_direction(
        X, Z, theta1_0, theta2_0, max_iter, tol,
        min_points_per_segment, min_points_to_fit,
        step_size, ridge_alpha, flip_split_direction=False
    )
    if res1 and res1[-1] < best_rmse:
        best_rmse, best_res, best_iters = res1[-1], res1, iters1

    res2, iters2 = _run_one_split_direction(
        X, Z, theta1_0, theta2_0, max_iter, tol,
        min_points_per_segment, min_points_to_fit,
        step_size, ridge_alpha, flip_split_direction=True
    )
    if res2 and res2[-1] < best_rmse:
        best_rmse, best_res, best_iters = res2[-1], res2, iters2

    if best_res is None:
        return None, None, None, None, None, max(iters1, iters2)

    split_coeffs, _, _, mask_left, mask_right, _best_rmse = best_res
    return split_coeffs, None, None, mask_left, mask_right, best_iters

def recursive_piecewise_fit_nd(
    X_full, Z_full, current_indices, threshold,
    min_points_per_segment=MIN_POINTS_FOR_SPLIT_BASE,
    depth=0, max_depth=5,
    current_run_random_state=None,
    step_size_optimize=0.1,
    ridge_alpha=0.0,
    iter_counts_list=None
) -> Node:
    if iter_counts_list is None:
        iter_counts_list = []

    X_node = X_full[current_indices]
    Z_node = Z_full[current_indices]
    n_points, n_features = X_node.shape[0], X_full.shape[1]
    min_points_to_fit = n_features + 1

    region = [(float(np.min(X_node[:, d])), float(np.max(X_node[:, d]))) for d in range(n_features)] if n_points > 0 else []
    mean_z = float(np.mean(Z_node)) if n_points > 0 else 0.0

    if n_points == 0:
        return Node(True, region, params=np.array([np.nan] * (n_features + 1)),
                    stop_reason="empty_segment", data_indices=current_indices)

    if n_points < min_points_to_fit:
        return Node(True, region, params=np.array([0.0] * n_features + [mean_z]),
                    stop_reason="not_enough_points_for_fit", data_indices=current_indices)

    theta_single = fit_single_plane(X_node, Z_node, ridge_alpha=ridge_alpha)
    if np.any(~np.isfinite(theta_single)):
        return Node(True, region, params=np.array([0.0] * n_features + [mean_z]),
                    stop_reason="fit_failed_numerically", data_indices=current_indices)

    rmse_single = rmse(Z_node, linear_model_nd(X_node, theta_single))

    if depth >= max_depth:
        return Node(True, region, params=theta_single, stop_reason="max_depth_reached", data_indices=current_indices)

    if threshold >= 0 and rmse_single <= threshold:
        return Node(True, region, params=theta_single, stop_reason="RMSE_threshold_met", data_indices=current_indices)

    if n_points < max(2 * min_points_to_fit, 2 * min_points_per_segment):
        return Node(True, region, params=theta_single, stop_reason="not_enough_points_for_split_pre_check", data_indices=current_indices)

    split_successful = False
    mask_left = mask_right = None
    split_coeffs = None

    for attempt in range(3):
        seed = (current_run_random_state or 0) + depth * 1000 + attempt
        res = optimize_split_optimized_nd(
            X_node, Z_node,
            min_points_per_segment=min_points_per_segment,
            step_size=step_size_optimize,
            random_seed_for_init=seed,
            ridge_alpha=ridge_alpha
        )
        if res and res[0] is not None:
            split_coeffs, _, _, mask_left, mask_right, iters = res
            if np.sum(mask_left) >= min_points_per_segment and np.sum(mask_right) >= min_points_per_segment:
                split_successful = True
                iter_counts_list.append(iters)
                break

    if not split_successful or split_coeffs is None or mask_left is None or mask_right is None:
        return Node(True, region, params=theta_single, stop_reason="split_failed", data_indices=current_indices)

    left = recursive_piecewise_fit_nd(
        X_full, Z_full, current_indices[mask_left], threshold,
        min_points_per_segment, depth + 1, max_depth,
        current_run_random_state, step_size_optimize, ridge_alpha, iter_counts_list
    )
    right = recursive_piecewise_fit_nd(
        X_full, Z_full, current_indices[mask_right], threshold,
        min_points_per_segment, depth + 1, max_depth,
        current_run_random_state, step_size_optimize, ridge_alpha, iter_counts_list
    )
    return Node(False, region, split_coeffs=split_coeffs, children=[left, right], data_indices=current_indices)
