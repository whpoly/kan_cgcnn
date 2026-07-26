from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _prediction_transform(
    raw: torch.Tensor,
    task_type: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> torch.Tensor:
    raw = raw.reshape(len(raw), -1)
    if task_type == "classification":
        return torch.sigmoid(raw)
    mean = torch.as_tensor(target_mean, dtype=raw.dtype, device=raw.device)
    std = torch.as_tensor(target_std, dtype=raw.dtype, device=raw.device)
    return raw * std + mean


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    task_type: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(
            np.asarray(features[start : start + batch_size], dtype=np.float32)
        ).to(device)
        predictions.append(
            _prediction_transform(
                model(batch),
                task_type,
                target_mean,
                target_std,
            )
            .detach()
            .cpu()
            .numpy()
        )
    if not predictions:
        return np.empty((0, len(np.asarray(target_mean).reshape(-1))), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def _metric(
    prediction: np.ndarray,
    target: np.ndarray,
    task_type: str,
) -> tuple[str, float, list[float]]:
    prediction = np.asarray(prediction, dtype=float).reshape(len(prediction), -1)
    target = np.asarray(target, dtype=float).reshape(len(target), -1)
    if task_type == "classification":
        labels = target[:, 0].astype(int)
        if len(np.unique(labels)) >= 2:
            value = float(roc_auc_score(labels, prediction[:, 0]))
            return "rocauc", value, [value]
        value = float(np.mean((prediction[:, 0] >= 0.5) == labels))
        return "accuracy", value, [value]
    per_target = np.mean(np.abs(prediction - target), axis=0)
    return "mae", float(np.mean(per_target)), per_target.astype(float).tolist()


def _gradient_importance(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    task_type: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n_features = features.shape[1]
    n_targets = len(np.asarray(target_mean).reshape(-1))
    aggregate = np.zeros(n_features, dtype=np.float64)
    per_target = np.zeros((n_targets, n_features), dtype=np.float64)
    total_rows = 0
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(
            np.asarray(features[start : start + batch_size], dtype=np.float32)
        ).to(device)
        batch.requires_grad_(True)
        prediction = _prediction_transform(
            model(batch),
            task_type,
            target_mean,
            target_std,
        )
        batch_target_gradients = []
        for target_index in range(prediction.shape[1]):
            gradient = torch.autograd.grad(
                prediction[:, target_index].sum(),
                batch,
                retain_graph=target_index + 1 < prediction.shape[1],
                create_graph=False,
            )[0]
            values = gradient.detach().abs().sum(dim=0).cpu().numpy()
            per_target[target_index] += values
            batch_target_gradients.append(values)
        aggregate += np.mean(batch_target_gradients, axis=0)
        total_rows += len(batch)
    denominator = max(total_rows, 1)
    return aggregate / denominator, per_target / denominator


def _raw_value(value: float, scale: float, offset: float) -> float:
    if abs(scale) <= 1e-15:
        return float("nan")
    return float((value - offset) / scale)


def _shape_label(values: Sequence[float]) -> str:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values) <= 1e-12:
        return "flat"
    positions = np.arange(len(values), dtype=float)
    correlation = float(np.corrcoef(positions, values)[0, 1])
    if correlation >= 0.8:
        return "increasing"
    if correlation <= -0.8:
        return "decreasing"
    middle = float(np.mean(values[len(values) // 3 : max(2 * len(values) // 3, 1)]))
    endpoints = float(np.mean([values[0], values[-1]]))
    amplitude = float(np.ptp(values))
    if endpoints - middle >= 0.25 * amplitude:
        return "u_shaped"
    if middle - endpoints >= 0.25 * amplitude:
        return "inverted_u"
    return "non_monotonic"


def _ale_curve(
    model: torch.nn.Module,
    features: np.ndarray,
    feature_index: int,
    device: torch.device,
    task_type: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
    bins: int,
    scale: float,
    offset: float,
) -> list[dict[str, Any]]:
    column = np.asarray(features[:, feature_index], dtype=float)
    edges = np.unique(np.quantile(column, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return []
    local_effects = []
    counts = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index + 1 == len(edges) - 1:
            mask = (column >= lower) & (column <= upper)
        else:
            mask = (column >= lower) & (column < upper)
        indices = np.flatnonzero(mask)
        if not len(indices):
            local_effects.append(np.zeros(len(np.asarray(target_mean).reshape(-1))))
            counts.append(0)
            continue
        lower_features = np.asarray(features[indices], dtype=np.float32).copy()
        upper_features = lower_features.copy()
        lower_features[:, feature_index] = lower
        upper_features[:, feature_index] = upper
        paired = np.concatenate([lower_features, upper_features], axis=0)
        prediction = _predict(
            model,
            paired,
            device,
            task_type,
            target_mean,
            target_std,
            batch_size,
        )
        lower_prediction = prediction[: len(indices)]
        upper_prediction = prediction[len(indices) :]
        local_effects.append(np.mean(upper_prediction - lower_prediction, axis=0))
        counts.append(len(indices))

    local = np.asarray(local_effects, dtype=float)
    cumulative = np.cumsum(local, axis=0) - 0.5 * local
    weights = np.asarray(counts, dtype=float)
    if weights.sum() > 0:
        cumulative -= np.average(cumulative, axis=0, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    records = []
    for bin_index, center in enumerate(centers):
        records.append(
            {
                "quantile": float((bin_index + 0.5) / len(centers)),
                "scaled_value": float(center),
                "raw_value": _raw_value(float(center), scale, offset),
                "count": int(counts[bin_index]),
                "effects": cumulative[bin_index].astype(float).tolist(),
            }
        )
    return records


def analyze_feature_patterns(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    device: torch.device,
    task_type: str,
    target_mean: np.ndarray | float,
    target_std: np.ndarray | float,
    feature_scales: Sequence[float],
    feature_offsets: Sequence[float],
    batch_size: int = 256,
    candidate_count: int = 32,
    top_k: int = 10,
    ale_bins: int = 10,
    permutation_repeats: int = 1,
    max_samples: int = 4096,
    seed: int = 7,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(len(features), -1)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("Feature matrix and feature names have incompatible widths")
    if len(features) != len(targets) or len(features) < 2:
        raise ValueError("Feature-pattern analysis needs at least two labelled samples")
    if targets.shape[1] != len(target_names):
        raise ValueError("Target matrix and target names have incompatible widths")
    if candidate_count < 1 or top_k < 1 or ale_bins < 2 or permutation_repeats < 1:
        raise ValueError("Feature-pattern analysis limits must be positive")
    if len(feature_scales) != features.shape[1] or len(feature_offsets) != features.shape[1]:
        raise ValueError("Feature preprocessing parameters do not match feature width")

    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(features) > max_samples:
        selected_rows = np.sort(rng.choice(len(features), size=max_samples, replace=False))
        features = features[selected_rows]
        targets = targets[selected_rows]

    target_mean_array = np.broadcast_to(
        np.asarray(target_mean, dtype=np.float32).reshape(-1),
        (targets.shape[1],),
    ).copy()
    target_std_array = np.broadcast_to(
        np.asarray(target_std, dtype=np.float32).reshape(-1),
        (targets.shape[1],),
    ).copy()
    baseline_prediction = _predict(
        model,
        features,
        device,
        task_type,
        target_mean_array,
        target_std_array,
        batch_size,
    )
    metric_name, baseline_metric, baseline_by_target = _metric(
        baseline_prediction,
        targets,
        task_type,
    )
    gradient, gradient_by_target = _gradient_importance(
        model,
        features,
        device,
        task_type,
        target_mean_array,
        target_std_array,
        batch_size,
    )
    candidate_count = min(candidate_count, features.shape[1])
    candidate_indices = np.argsort(-gradient)[:candidate_count].astype(int).tolist()

    records = []
    for feature_index in candidate_indices:
        permuted_metrics = []
        permuted_by_target = []
        for _ in range(permutation_repeats):
            permuted = features.copy()
            permuted[:, feature_index] = permuted[
                rng.permutation(len(permuted)), feature_index
            ]
            prediction = _predict(
                model,
                permuted,
                device,
                task_type,
                target_mean_array,
                target_std_array,
                batch_size,
            )
            _, metric_value, target_values = _metric(prediction, targets, task_type)
            permuted_metrics.append(metric_value)
            permuted_by_target.append(target_values)
        permuted_metric = float(np.mean(permuted_metrics))
        per_target_metric = np.mean(np.asarray(permuted_by_target), axis=0)
        if metric_name in {"rocauc", "accuracy"}:
            importance = baseline_metric - permuted_metric
            per_target_importance = (
                np.asarray(baseline_by_target) - per_target_metric
            )
        else:
            importance = permuted_metric - baseline_metric
            per_target_importance = (
                per_target_metric - np.asarray(baseline_by_target)
            )
        records.append(
            {
                "feature": str(feature_names[feature_index]),
                "feature_index": int(feature_index),
                "gradient_importance": float(gradient[feature_index]),
                "gradient_importance_by_target": gradient_by_target[
                    :, feature_index
                ]
                .astype(float)
                .tolist(),
                "permutation_importance": float(importance),
                "permutation_importance_by_target": per_target_importance.astype(
                    float
                ).tolist(),
                "permuted_metric": permuted_metric,
            }
        )

    records.sort(
        key=lambda item: (
            -float(item["permutation_importance"]),
            -float(item["gradient_importance"]),
            str(item["feature"]),
        )
    )
    selected_records = records[: min(top_k, len(records))]
    for rank, record in enumerate(records, start=1):
        record["permutation_rank"] = rank
        record["selected_for_ale"] = rank <= len(selected_records)
        record["ale"] = {}
    for record in selected_records:
        curve = _ale_curve(
            model,
            features,
            int(record["feature_index"]),
            device,
            task_type,
            target_mean_array,
            target_std_array,
            batch_size,
            ale_bins,
            float(feature_scales[int(record["feature_index"])]),
            float(feature_offsets[int(record["feature_index"])]),
        )
        for target_index, target_name in enumerate(target_names):
            effects = [float(point["effects"][target_index]) for point in curve]
            record["ale"][str(target_name)] = {
                "shape": _shape_label(effects),
                "points": [
                    {
                        key: value
                        for key, value in point.items()
                        if key != "effects"
                    }
                    | {"effect": effects[point_index]}
                    for point_index, point in enumerate(curve)
                ],
            }

    return {
        "method": "heldout_gradient_permutation_ale",
        "task_type": task_type,
        "metric": metric_name,
        "baseline_metric": baseline_metric,
        "baseline_metric_by_target": baseline_by_target,
        "n_samples": len(features),
        "n_features": features.shape[1],
        "candidate_count": candidate_count,
        "top_k": len(selected_records),
        "ale_bins": ale_bins,
        "permutation_repeats": permutation_repeats,
        "feature_names": [str(name) for name in feature_names],
        "target_names": [str(name) for name in target_names],
        "features": records,
    }


def pairwise_curve_correlation(curves: Sequence[Sequence[float]]) -> float | None:
    usable = [np.asarray(curve, dtype=float) for curve in curves if len(curve) >= 2]
    values = []
    for left, right in combinations(usable, 2):
        size = min(len(left), len(right))
        grid = np.linspace(0.0, 1.0, size)
        left_interp = np.interp(
            grid, np.linspace(0.0, 1.0, len(left)), left
        )
        right_interp = np.interp(
            grid, np.linspace(0.0, 1.0, len(right)), right
        )
        if np.std(left_interp) <= 1e-12 or np.std(right_interp) <= 1e-12:
            continue
        values.append(float(np.corrcoef(left_interp, right_interp)[0, 1]))
    return float(np.mean(values)) if values else None
