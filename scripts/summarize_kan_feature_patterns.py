from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgcnn_pyg_kan.feature_patterns import pairwise_curve_correlation
from cgcnn_pyg_kan.spline_symbolic import (
    SPLINE_SYMBOLIC_FUNCTIONS,
    fit_edge_function,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize held-out feature importance and ALE patterns shared by "
            "multiple five-fold KAN teachers."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pattern-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["fastkan", "spline"])
    parser.add_argument("--comparison-jsons", nargs="+", required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--min-fold-frequency", type=int, default=4)
    parser.add_argument(
        "--min-curve-correlation",
        type=float,
        default=0.6,
        help=(
            "Minimum within-model cross-fold and cross-model ALE correlation "
            "for a performance-qualified consensus pattern."
        ),
    )
    parser.add_argument(
        "--min-permutation-importance",
        type=float,
        default=0.0,
        help=(
            "Every model must have mean held-out permutation importance strictly "
            "above this value."
        ),
    )
    parser.add_argument(
        "--derived-feature-functions",
        nargs="+",
        choices=SPLINE_SYMBOLIC_FUNCTIONS,
        default=[
            "identity",
            "square",
            "cube",
            "tanh",
            "log",
            "sqrt",
            "reciprocal",
            "abs",
            "arctan",
            "gaussian",
        ],
    )
    parser.add_argument("--derived-feature-min-r2", type=float, default=0.8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _curve(record: dict[str, Any], target: str) -> list[float]:
    ale = record.get("ale", {}).get(target, {})
    return [float(point["effect"]) for point in ale.get("points", [])]


def _raw_curve(record: dict[str, Any], target: str) -> list[float]:
    ale = record.get("ale", {}).get(target, {})
    return [float(point["raw_value"]) for point in ale.get("points", [])]


def _curve_on_grid(curve: list[float], size: int = 21) -> np.ndarray | None:
    if len(curve) < 2:
        return None
    values = np.asarray(curve, dtype=float)
    return np.interp(
        np.linspace(0.0, 1.0, size),
        np.linspace(0.0, 1.0, len(values)),
        values,
    )


def _correlation(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _model_feature_summary(
    payloads: list[dict[str, Any]],
    feature: str,
    target: str,
) -> dict[str, Any]:
    selected_records = []
    importance_by_fold = []
    curves = []
    raw_curves = []
    shapes = Counter()
    for payload in payloads:
        by_name = {
            str(record["feature"]): record
            for record in payload.get("features", [])
        }
        record = by_name.get(feature)
        importance_by_fold.append(
            float(record["permutation_importance"]) if record is not None else 0.0
        )
        if record is None or not bool(record.get("selected_for_ale", False)):
            continue
        selected_records.append(record)
        curve = _curve(record, target)
        if curve:
            curves.append(curve)
            raw_curves.append(_raw_curve(record, target))
        shape = record.get("ale", {}).get(target, {}).get("shape")
        if shape:
            shapes[str(shape)] += 1
    dominant_shape = (
        sorted(shapes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if shapes
        else ""
    )
    interpolated_curves = [
        item for curve in curves if (item := _curve_on_grid(curve)) is not None
    ]
    averaged_curve = (
        np.mean(interpolated_curves, axis=0) if interpolated_curves else None
    )
    std_curve = (
        np.std(interpolated_curves, axis=0) if interpolated_curves else None
    )
    interpolated_raw_curves = [
        item for curve in raw_curves if (item := _curve_on_grid(curve)) is not None
    ]
    averaged_raw_curve = (
        np.mean(interpolated_raw_curves, axis=0)
        if interpolated_raw_curves
        else None
    )
    return {
        "fold_frequency": len(selected_records),
        "mean_permutation_importance": float(mean(importance_by_fold)),
        "std_permutation_importance": (
            float(pstdev(importance_by_fold)) if len(importance_by_fold) > 1 else 0.0
        ),
        "within_model_curve_correlation": pairwise_curve_correlation(curves),
        "dominant_shape": dominant_shape,
        "shape_frequency": dict(sorted(shapes.items())),
        "mean_curve": averaged_curve.astype(float).tolist() if averaged_curve is not None else None,
        "std_curve": std_curve.astype(float).tolist() if std_curve is not None else None,
        "mean_raw_curve": (
            averaged_raw_curve.astype(float).tolist()
            if averaged_raw_curve is not None
            else None
        ),
    }


def _primitive_expression(function: str, argument: str) -> str:
    return {
        "identity": f"({argument})",
        "square": f"({argument})^2",
        "cube": f"({argument})^3",
        "sin": f"sin({argument})",
        "cos": f"cos({argument})",
        "tan": f"protected_tan({argument})",
        "tanh": f"tanh({argument})",
        "exp": f"protected_exp({argument})",
        "log": f"log(abs({argument})+1e-3)",
        "sqrt": f"signed_sqrt({argument})",
        "reciprocal": f"protected_reciprocal({argument}; eps=1e-3)",
        "abs": f"abs({argument})",
        "arctan": f"arctan({argument})",
        "gaussian": f"exp(-({argument})^2)",
    }[function]


def _derive_feature(
    feature: str,
    target: str,
    by_model: dict[str, dict[str, Any]],
    functions: list[str],
    min_r2: float,
) -> dict[str, Any] | None:
    raw_curves = [
        np.asarray(item["mean_raw_curve"], dtype=float)
        for item in by_model.values()
        if item["mean_raw_curve"] is not None
    ]
    effect_curves = [
        np.asarray(item["mean_curve"], dtype=float)
        for item in by_model.values()
        if item["mean_curve"] is not None
    ]
    if not raw_curves or not effect_curves:
        return None
    raw = np.mean(raw_curves, axis=0)
    effect = np.mean(effect_curves, axis=0)
    finite = np.isfinite(raw) & np.isfinite(effect)
    raw = raw[finite]
    effect = effect[finite]
    if len(raw) < 3 or np.std(raw) <= 1e-12 or np.std(effect) <= 1e-12:
        return None
    center = float(np.mean(raw))
    scale = float(np.std(raw))
    normalized = (raw - center) / scale
    specification = fit_edge_function(
        normalized,
        effect,
        functions,
        search_range=5.0,
        grid_size=21,
        iterations=2,
        complexity_weight=0.1,
        epsilon=1e-3,
    )
    raw_a = float(specification["a"]) / scale
    raw_b = float(specification["b"]) - float(specification["a"]) * center / scale
    function = str(specification["function"])
    argument = f"{raw_a:.10g}*raw({feature!r}){raw_b:+.10g}"
    primitive = _primitive_expression(function, argument)
    symbolic_r2 = float(specification["r2"])
    safe_feature = _safe_filename(feature)
    if symbolic_r2 >= min_r2:
        return {
            "name": f"kan_consensus__{function}__{safe_feature}",
            "source_features": [feature],
            "target": target,
            "kind": "symbolic_unary",
            "novel_transform": function != "identity",
            "function": function,
            "a": raw_a,
            "b": raw_b,
            "expression": primitive,
            "response_scale": float(specification["c"]),
            "response_offset": float(specification["d"]),
            "response_expression": (
                f"{float(specification['c']):.10g}*derived_feature"
                f"{float(specification['d']):+.10g}"
            ),
            "pattern_fit_r2": symbolic_r2,
            "raw_knots": raw.astype(float).tolist(),
            "ale_effect_knots": effect.astype(float).tolist(),
        }
    return {
        "name": f"kan_consensus__piecewise__{safe_feature}",
        "source_features": [feature],
        "target": target,
        "kind": "piecewise_linear_ale",
        "novel_transform": True,
        "function": "piecewise_linear",
        "expression": (
            f"linear_interp(raw({feature!r}), raw_knots, ale_effect_knots)"
        ),
        "response_scale": 1.0,
        "response_offset": 0.0,
        "response_expression": "derived_feature",
        "pattern_fit_r2": 1.0,
        "best_symbolic_function": function,
        "best_symbolic_r2": symbolic_r2,
        "raw_knots": raw.astype(float).tolist(),
        "ale_effect_knots": effect.astype(float).tolist(),
    }


def summarize(
    dataset: str,
    pattern_dir: Path,
    models: list[str],
    comparison_paths: list[Path],
    expected_folds: int,
    min_fold_frequency: int,
    min_curve_correlation: float,
    min_permutation_importance: float,
    derived_feature_functions: list[str],
    derived_feature_min_r2: float,
) -> dict[str, Any]:
    comparisons = {}
    for path in comparison_paths:
        payload = _load_json(path)
        model = str(payload.get("kan_model", ""))
        if not model:
            raise ValueError(f"Comparison JSON has no kan_model: {path}")
        comparisons[model] = payload
    missing_comparisons = sorted(set(models) - set(comparisons))
    if missing_comparisons:
        raise ValueError(
            f"Missing performance comparisons for models: {missing_comparisons}"
        )

    patterns: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        payloads = []
        for fold in range(expected_folds):
            path = pattern_dir / (
                f"feature-patterns-{dataset}-fold{fold}-{model}.json"
            )
            payload = _load_json(path)
            if int(payload.get("fold", -1)) != fold or payload.get("model") != model:
                raise ValueError(f"Feature-pattern identity mismatch: {path}")
            payloads.append(payload)
        patterns[model] = payloads

    target_names = sorted(
        {
            str(target)
            for payloads in patterns.values()
            for payload in payloads
            for target in payload.get("target_names", [])
        }
    )
    feature_names = sorted(
        {
            str(record["feature"])
            for payloads in patterns.values()
            for payload in payloads
            for record in payload.get("features", [])
            if bool(record.get("selected_for_ale", False))
        }
    )
    performance = {
        model: {
            "passes_teacher_closeness_gate": bool(
                comparisons[model].get("passes_teacher_closeness_gate", False)
            ),
            "teacher_mean": comparisons[model].get("teacher_mean"),
            "official_mean": comparisons[model].get("official_mean"),
            "teacher_mean_degradation_pct": comparisons[model].get(
                "teacher_mean_degradation_pct"
            ),
            "teacher_fold_wins": comparisons[model].get("teacher_fold_wins"),
        }
        for model in models
    }
    all_models_performance_qualified = all(
        item["passes_teacher_closeness_gate"] for item in performance.values()
    )

    rows = []
    pattern_details = []
    for target in target_names:
        for feature in feature_names:
            by_model = {
                model: _model_feature_summary(patterns[model], feature, target)
                for model in models
            }
            model_curves = [
                np.asarray(item["mean_curve"], dtype=float)
                if item["mean_curve"] is not None
                else None
                for item in by_model.values()
            ]
            cross_correlations = [
                value
                for left_index, left in enumerate(model_curves)
                for right in model_curves[left_index + 1 :]
                if (value := _correlation(left, right)) is not None
            ]
            cross_model_curve_correlation = (
                float(mean(cross_correlations)) if cross_correlations else None
            )
            frequency_passes = all(
                item["fold_frequency"] >= min_fold_frequency
                for item in by_model.values()
            )
            importance_passes = all(
                item["mean_permutation_importance"]
                > min_permutation_importance
                for item in by_model.values()
            )
            within_curve_passes = all(
                item["within_model_curve_correlation"] is not None
                and item["within_model_curve_correlation"]
                >= min_curve_correlation
                for item in by_model.values()
            )
            cross_curve_passes = (
                cross_model_curve_correlation is not None
                and cross_model_curve_correlation >= min_curve_correlation
            )
            curve_passes = within_curve_passes and cross_curve_passes
            consensus = (
                all_models_performance_qualified
                and frequency_passes
                and importance_passes
                and curve_passes
            )
            derived_feature = (
                _derive_feature(
                    feature,
                    target,
                    by_model,
                    derived_feature_functions,
                    derived_feature_min_r2,
                )
                if consensus
                else None
            )
            if derived_feature is not None:
                derived_feature["evidence"] = {
                    "cross_model_curve_correlation": (
                        cross_model_curve_correlation
                    ),
                    "models": {
                        model: {
                            "fold_frequency": item["fold_frequency"],
                            "mean_permutation_importance": item[
                                "mean_permutation_importance"
                            ],
                            "within_model_curve_correlation": item[
                                "within_model_curve_correlation"
                            ],
                            "dominant_shape": item["dominant_shape"],
                        }
                        for model, item in by_model.items()
                    },
                }
            pattern_details.append(
                {
                    "target": target,
                    "feature": feature,
                    "consensus_pattern": consensus,
                    "cross_model_curve_correlation": (
                        cross_model_curve_correlation
                    ),
                    "models": by_model,
                    "derived_feature": derived_feature,
                }
            )
            row: dict[str, Any] = {
                "dataset": dataset,
                "target": target,
                "feature": feature,
                "performance_qualified": all_models_performance_qualified,
                "frequency_qualified": frequency_passes,
                "importance_qualified": importance_passes,
                "within_curve_qualified": within_curve_passes,
                "cross_model_curve_correlation": cross_model_curve_correlation,
                "cross_curve_qualified": cross_curve_passes,
                "curve_qualified": curve_passes,
                "consensus_pattern": consensus,
                "derived_feature_name": (
                    derived_feature["name"] if derived_feature else ""
                ),
                "derived_feature_kind": (
                    derived_feature["kind"] if derived_feature else ""
                ),
                "derived_feature_expression": (
                    derived_feature["expression"] if derived_feature else ""
                ),
                "derived_feature_pattern_fit_r2": (
                    derived_feature["pattern_fit_r2"]
                    if derived_feature
                    else None
                ),
                "novel_transform": (
                    derived_feature["novel_transform"]
                    if derived_feature
                    else False
                ),
            }
            for model, item in by_model.items():
                prefix = model.replace("-", "_")
                row[f"{prefix}_fold_frequency"] = item["fold_frequency"]
                row[f"{prefix}_mean_permutation_importance"] = item[
                    "mean_permutation_importance"
                ]
                row[f"{prefix}_std_permutation_importance"] = item[
                    "std_permutation_importance"
                ]
                row[f"{prefix}_within_curve_correlation"] = item[
                    "within_model_curve_correlation"
                ]
                row[f"{prefix}_dominant_shape"] = item["dominant_shape"]
            rows.append(row)

    rows.sort(
        key=lambda row: (
            not bool(row["consensus_pattern"]),
            -min(
                int(row[f"{model.replace('-', '_')}_fold_frequency"])
                for model in models
            ),
            -min(
                float(row[f"{model.replace('-', '_')}_mean_permutation_importance"])
                for model in models
            ),
            str(row["target"]),
            str(row["feature"]),
        )
    )
    consensus_rows = [row for row in rows if row["consensus_pattern"]]
    return {
        "dataset": dataset,
        "models": models,
        "expected_folds": expected_folds,
        "min_fold_frequency": min_fold_frequency,
        "min_curve_correlation": min_curve_correlation,
        "min_permutation_importance": min_permutation_importance,
        "derived_feature_functions": derived_feature_functions,
        "derived_feature_min_r2": derived_feature_min_r2,
        "performance": performance,
        "all_models_performance_qualified": all_models_performance_qualified,
        "consensus_feature_count": len(consensus_rows),
        "consensus_features": [
            {"target": row["target"], "feature": row["feature"]}
            for row in consensus_rows
        ],
        "derived_features": [
            detail["derived_feature"]
            for detail in pattern_details
            if detail["derived_feature"] is not None
        ],
        "pattern_details": pattern_details,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _derived_feature_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in payload["derived_features"]:
        rows.append(
            {
                "dataset": payload["dataset"],
                "target": feature["target"],
                "name": feature["name"],
                "source_features": ",".join(feature["source_features"]),
                "kind": feature["kind"],
                "novel_transform": feature["novel_transform"],
                "function": feature["function"],
                "expression": feature["expression"],
                "response_expression": feature["response_expression"],
                "pattern_fit_r2": feature["pattern_fit_r2"],
                "best_symbolic_function": feature.get(
                    "best_symbolic_function", feature["function"]
                ),
                "best_symbolic_r2": feature.get(
                    "best_symbolic_r2", feature["pattern_fit_r2"]
                ),
                "raw_knots": json.dumps(feature["raw_knots"]),
                "ale_effect_knots": json.dumps(feature["ale_effect_knots"]),
            }
        )
    return rows


def _report(payload: dict[str, Any]) -> str:
    lines = [
        f"FastKAN/Spline-KAN feature-pattern consensus: {payload['dataset']}",
        "",
        "Performance gate:",
    ]
    for model, item in payload["performance"].items():
        lines.append(
            f"  {model}: qualified={item['passes_teacher_closeness_gate']}, "
            f"official={item['official_mean']}, teacher={item['teacher_mean']}, "
            f"degradation={item['teacher_mean_degradation_pct']}%"
        )
    lines.extend(
        [
            "",
            "Consensus rule:",
            f"  selected for ALE in >= {payload['min_fold_frequency']}/{payload['expected_folds']} folds for every model",
            f"  mean held-out permutation importance > {payload['min_permutation_importance']} for every model",
            f"  within-model cross-fold and cross-model ALE correlation >= {payload['min_curve_correlation']}",
            "  every model passes its performance-closeness gate",
            "",
            f"Consensus patterns ({payload['consensus_feature_count']}):",
        ]
    )
    for row in payload["rows"]:
        if not row["consensus_pattern"]:
            continue
        model_parts = []
        for model in payload["models"]:
            prefix = model.replace("-", "_")
            model_parts.append(
                f"{model}={row[f'{prefix}_fold_frequency']}/{payload['expected_folds']} "
                f"({row[f'{prefix}_dominant_shape']}, "
                f"importance={row[f'{prefix}_mean_permutation_importance']:.6g})"
            )
        lines.append(
            f"  {row['target']} :: {row['feature']} | "
            + "; ".join(model_parts)
            + f"; cross-model curve r={row['cross_model_curve_correlation']:.6g}"
        )
        lines.append(
            f"    derived feature = {row['derived_feature_name']}"
        )
        lines.append(
            f"    expression = {row['derived_feature_expression']}"
        )
        lines.append(
            "    pattern fit R2 = "
            f"{row['derived_feature_pattern_fit_r2']:.6g}; "
            f"novel transform = {row['novel_transform']}"
        )
    if payload["consensus_feature_count"] == 0:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Interpretation note:",
            "  Importance and ALE are descriptive held-out associations, not causal effects.",
            "  Outer-test explanations were not used for training or performance selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _safe_filename(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    ).strip("_")
    return safe[:120] or "feature"


def _write_consensus_plots(
    payload: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        print(
            "matplotlib is unavailable; skipped consensus ALE plots",
            flush=True,
        )
        return []

    paths = []
    for detail in payload["pattern_details"]:
        if not detail["consensus_pattern"]:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        for model in payload["models"]:
            item = detail["models"][model]
            curve = item["mean_curve"]
            if curve is None:
                continue
            curve_array = np.asarray(curve, dtype=float)
            std_array = np.asarray(item["std_curve"], dtype=float)
            raw_values = np.asarray(item["mean_raw_curve"], dtype=float)
            axis.plot(
                raw_values,
                curve_array,
                linewidth=2.0,
                label=(
                    f"{model} ({item['fold_frequency']}/"
                    f"{payload['expected_folds']} folds)"
                ),
            )
            axis.fill_between(
                raw_values,
                curve_array - std_array,
                curve_array + std_array,
                alpha=0.18,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xlabel(str(detail["feature"]))
        axis.set_ylabel(f"ALE effect on {detail['target']}")
        axis.set_title(
            f"{payload['dataset']}: FastKAN/Spline-KAN shared response\n"
            f"cross-model curve r={detail['cross_model_curve_correlation']:.3f}"
        )
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        filename = (
            "ale-consensus-"
            f"{_safe_filename(str(detail['target']))}-"
            f"{_safe_filename(str(detail['feature']))}.png"
        )
        path = output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.expected_folds < 2:
        raise ValueError("--expected-folds must be at least 2")
    if not 1 <= args.min_fold_frequency <= args.expected_folds:
        raise ValueError("--min-fold-frequency must be between 1 and expected folds")
    if not -1.0 <= args.min_curve_correlation <= 1.0:
        raise ValueError("--min-curve-correlation must be in [-1, 1]")
    if not 0.0 <= args.derived_feature_min_r2 <= 1.0:
        raise ValueError("--derived-feature-min-r2 must be in [0, 1]")
    payload = summarize(
        args.dataset,
        Path(args.pattern_dir),
        list(args.models),
        [Path(path) for path in args.comparison_jsons],
        args.expected_folds,
        args.min_fold_frequency,
        args.min_curve_correlation,
        args.min_permutation_importance,
        list(args.derived_feature_functions),
        args.derived_feature_min_r2,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kan-feature-consensus-{args.dataset}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    report_path = output_dir / f"{stem}.txt"
    derived_csv_path = output_dir / f"derived-features-{args.dataset}.csv"
    derived_json_path = output_dir / f"derived-features-{args.dataset}.json"
    _write_csv(csv_path, payload["rows"])
    payload["plot_paths"] = _write_consensus_plots(payload, output_dir)
    _write_csv(derived_csv_path, _derived_feature_rows(payload))
    derived_json_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "generation_method": (
                    "performance-qualified cross-fold FastKAN/Spline-KAN "
                    "consensus ALE transformation"
                ),
                "features": payload["derived_features"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(_report(payload), encoding="utf-8")
    print(
        f"{args.dataset}: {payload['consensus_feature_count']} performance-qualified "
        f"FastKAN/Spline-KAN consensus patterns",
        flush=True,
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(f"Wrote {derived_csv_path}", flush=True)
    print(f"Wrote {derived_json_path}", flush=True)


if __name__ == "__main__":
    main()
