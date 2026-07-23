from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed five-fold KAN teacher or symbolic-formula results "
            "with completed official MODNet folds."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--official-fold-csv", required=True)
    parser.add_argument("--kan-output-dir", required=True)
    parser.add_argument("--kan-fold-csv", default=None)
    parser.add_argument("--model", default="fastkan")
    parser.add_argument(
        "--comparison-target",
        choices=["teacher", "formula", "spline-symbolic", "symbolic-kan"],
        default="teacher",
    )
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.02,
        help="Required mean relative MAE reduction for regression.",
    )
    parser.add_argument(
        "--min-absolute-improvement",
        type=float,
        default=0.0,
        help="Required mean absolute ROC-AUC increase for classification.",
    )
    parser.add_argument("--min-fold-wins", type=int, default=3)
    parser.add_argument(
        "--min-formula-fidelity-r2",
        type=float,
        default=0.90,
        help="Required mean formula-to-KAN teacher R2, expressed on a 0..1 scale.",
    )
    parser.add_argument(
        "--min-feature-jaccard",
        type=float,
        default=0.40,
        help="Required mean pairwise Jaccard similarity of active formula inputs.",
    )
    parser.add_argument(
        "--min-operator-jaccard",
        type=float,
        default=0.40,
        help="Required mean pairwise Jaccard similarity of selected symbolic operators.",
    )
    parser.add_argument(
        "--max-improvement-std",
        type=float,
        default=0.10,
        help="Maximum population standard deviation of fold performance improvements.",
    )
    parser.add_argument(
        "--max-teacher-relative-degradation",
        type=float,
        default=0.05,
        help="Maximum allowed mean MAE degradation of the KAN teacher versus official MODNet.",
    )
    parser.add_argument(
        "--max-formula-relative-degradation",
        type=float,
        default=0.05,
        help="Maximum allowed mean MAE degradation of the formula versus official MODNet.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def latest_kan_csv(directory: Path, dataset: str) -> Path:
    matches = list(directory.glob(f"modnet-kan-{dataset}-*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No modnet-kan-{dataset}-*.csv found under {directory}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def rows_by_fold(
    rows: list[dict[str, str]],
    metric: str,
    source: str,
    model: str | None = None,
) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        raw_fold = row.get("fold")
        if raw_fold is None or raw_fold == "":
            continue
        fold = int(raw_fold)
        finite_float(row.get(metric), f"{source} fold {fold} {metric}")
        if fold in selected:
            raise ValueError(f"Duplicate {source} row for fold {fold}")
        selected[fold] = row
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_artifact_path(raw_path: str, output_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    local_candidate = output_dir / candidate.name
    if local_candidate.is_file():
        return local_candidate
    raise FileNotFoundError(
        f"Could not resolve formula artifact {raw_path!r} from {output_dir}"
    )


def active_features(record: dict[str, Any]) -> list[str]:
    names = record.get("active_feature_names")
    if isinstance(names, list):
        return [str(name) for name in names]

    definitions = record.get("variable_definitions", [])
    active_variables = record.get("active_variable_indices")
    if isinstance(definitions, list) and definitions:
        if isinstance(active_variables, list):
            selected = []
            for index in active_variables:
                if 0 <= int(index) < len(definitions):
                    selected.append(str(definitions[int(index)]["feature"]))
            if selected:
                return selected
        return [str(item["feature"]) for item in definitions if "feature" in item]

    feature_names = record.get("feature_names", [])
    if isinstance(feature_names, list):
        return [str(name) for name in feature_names]
    return []


def selected_operators(term_names: list[str]) -> set[str]:
    operators: set[str] = set()
    for term in term_names:
        if term.startswith("sin("):
            operators.add("sin")
        elif term.startswith("cos("):
            operators.add("cos")
        elif term.startswith("tanh("):
            operators.add("tanh")
        elif term.startswith("exp("):
            operators.add("exp")
        elif term.startswith("log("):
            operators.add("log")
        elif term.startswith("sqrt("):
            operators.add("sqrt")
        elif term.startswith("1/protected("):
            operators.add("reciprocal")
        elif "/protected(" in term:
            operators.add("ratio")
        elif "*" in term:
            operators.add("product")
        elif "^3" in term:
            operators.add("cube")
        elif "^2" in term:
            operators.add("square")
        else:
            operators.add("identity")
    return operators


def load_fold_formulas(
    rows: dict[int, dict[str, str]],
    output_dir: Path,
    comparison_target: str = "formula",
) -> tuple[list[dict[str, Any]], dict[str, list[set[str]]]]:
    formulas: list[dict[str, Any]] = []
    feature_sets: dict[str, list[set[str]]] = defaultdict(list)
    for fold, row in sorted(rows.items()):
        path_field = {
            "formula": "simple_formula_json_path",
            "spline-symbolic": "spline_symbolic_json_path",
            "symbolic-kan": "symbolic_kan_json_path",
        }[comparison_target]
        raw_path = row.get(path_field, "")
        if not raw_path:
            raise ValueError(f"KAN fold {fold} has no {path_field}")
        path = resolve_artifact_path(raw_path, output_dir)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("targets")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Formula JSON has no targets: {path}")
        for record in records:
            target = str(record.get("target", "target"))
            features = active_features(record)
            term_names = [str(name) for name in record.get("term_names", [])]
            operators = record.get("operators")
            selected = (
                {str(name) for name in operators}
                if isinstance(operators, list)
                else selected_operators(term_names)
            )
            feature_sets[target].append(set(features))
            formulas.append(
                {
                    "fold": fold,
                    "target": target,
                    "features": features,
                    "term_names": term_names,
                    "operators": sorted(selected),
                    "variable_definitions": record.get("variable_definitions", []),
                    "hidden_definitions": record.get("hidden_definitions", []),
                    "expression": str(record.get("expression", "")),
                    "test_target_mae": record.get("test_target_mae"),
                    "test_teacher_mae": record.get("test_teacher_mae"),
                    "test_fidelity_r2": record.get("test_fidelity_r2"),
                    "formula_json_path": str(path),
                }
            )
    return formulas, feature_sets


def pairwise_jaccard(feature_sets: list[set[str]]) -> list[float]:
    values = []
    for left_index, left in enumerate(feature_sets):
        for right in feature_sets[left_index + 1 :]:
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return values


def formula_stability(
    formulas: list[dict[str, Any]],
    feature_sets: dict[str, list[set[str]]],
    expected_folds: int,
) -> dict[str, Any]:
    target_stability = {}
    all_jaccards = []
    all_operator_jaccards = []
    for target, sets in sorted(feature_sets.items()):
        if len(sets) != expected_folds:
            raise ValueError(
                f"Target {target!r} has formulas for {len(sets)} folds; "
                f"expected {expected_folds}"
            )
        jaccards = pairwise_jaccard(sets)
        all_jaccards.extend(jaccards)
        target_formulas = [
            item for item in formulas if item["target"] == target
        ]
        operator_sets = [set(item["operators"]) for item in target_formulas]
        operator_jaccards = pairwise_jaccard(operator_sets)
        all_operator_jaccards.extend(operator_jaccards)
        counts = Counter(feature for features in sets for feature in features)
        operator_counts = Counter(
            operator
            for operators in operator_sets
            for operator in operators
        )
        target_stability[target] = {
            "mean_pairwise_feature_jaccard": mean(jaccards) if jaccards else 1.0,
            "min_pairwise_feature_jaccard": min(jaccards) if jaccards else 1.0,
            "feature_frequency": dict(
                sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "features_in_at_least_4_folds": sorted(
                feature for feature, count in counts.items() if count >= 4
            ),
            "mean_pairwise_operator_jaccard": (
                mean(operator_jaccards) if operator_jaccards else 1.0
            ),
            "min_pairwise_operator_jaccard": (
                min(operator_jaccards) if operator_jaccards else 1.0
            ),
            "operator_fold_frequency": dict(
                sorted(
                    operator_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }

    fidelities = [
        finite_float(item["test_fidelity_r2"], "formula test_fidelity_r2")
        for item in formulas
        if item.get("test_fidelity_r2") is not None
    ]
    return {
        "mean_pairwise_feature_jaccard": mean(all_jaccards) if all_jaccards else 1.0,
        "min_pairwise_feature_jaccard": min(all_jaccards) if all_jaccards else 1.0,
        "mean_pairwise_operator_jaccard": (
            mean(all_operator_jaccards) if all_operator_jaccards else 1.0
        ),
        "min_pairwise_operator_jaccard": (
            min(all_operator_jaccards) if all_operator_jaccards else 1.0
        ),
        "mean_formula_teacher_fidelity_r2": mean(fidelities) if fidelities else None,
        "min_formula_teacher_fidelity_r2": min(fidelities) if fidelities else None,
        "targets": target_stability,
    }


def formula_report_lines(
    dataset: str,
    formulas: list[dict[str, Any]],
    stability: dict[str, Any],
) -> list[str]:
    lines = [
        f"Five-fold symbolic formula report: {dataset}",
        "",
        "Formula scope:",
        "  Each formula is learned without using its outer test fold.",
        "  x/z variables are fold-specific preprocessed official MODNet descriptors.",
        "  Fold formulas are intentionally reported separately; coefficients must not be averaged.",
        "",
        "Cross-fold stability:",
        f"  mean pairwise feature Jaccard = {stability['mean_pairwise_feature_jaccard']:.6g}",
        f"  min pairwise feature Jaccard = {stability['min_pairwise_feature_jaccard']:.6g}",
        f"  mean pairwise operator Jaccard = {stability['mean_pairwise_operator_jaccard']:.6g}",
        f"  min pairwise operator Jaccard = {stability['min_pairwise_operator_jaccard']:.6g}",
        f"  mean formula-to-teacher R2 = {stability['mean_formula_teacher_fidelity_r2']:.6g}",
        f"  min formula-to-teacher R2 = {stability['min_formula_teacher_fidelity_r2']:.6g}",
        "",
    ]
    for target, item in stability["targets"].items():
        lines.append(f"Consensus descriptors for target = {target}")
        for feature, count in item["feature_frequency"].items():
            lines.append(f"  {feature}: {count}/5 folds")
        lines.append("Selected symbolic operators:")
        for operator, count in item["operator_fold_frequency"].items():
            lines.append(f"  {operator}: {count}/5 folds")
        lines.append("")

    for item in formulas:
        lines.append(f"fold = {item['fold']}, target = {item['target']}")
        for definition in item["variable_definitions"]:
            lines.append(
                f"  {definition.get('variable')} = {definition.get('expression')}"
            )
        for definition in item.get("hidden_definitions", []):
            lines.append(
                f"  {definition.get('variable')} = {definition.get('expression')}"
            )
        lines.append(f"  formula = {item['expression']}")
        lines.append(f"  active descriptors = {','.join(item['features'])}")
        lines.append(f"  formula-to-target test MAE = {item['test_target_mae']}")
        lines.append(f"  formula-to-teacher test R2 = {item['test_fidelity_r2']}")
        lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    if args.expected_folds < 1:
        raise ValueError("--expected-folds must be positive")
    if not 0 <= args.min_fold_wins <= args.expected_folds:
        raise ValueError("--min-fold-wins must be between 0 and --expected-folds")
    if not 0 <= args.min_formula_fidelity_r2 <= 1:
        raise ValueError("--min-formula-fidelity-r2 must be in [0, 1]")
    if not 0 <= args.min_feature_jaccard <= 1:
        raise ValueError("--min-feature-jaccard must be in [0, 1]")
    if not 0 <= args.min_operator_jaccard <= 1:
        raise ValueError("--min-operator-jaccard must be in [0, 1]")
    if args.max_improvement_std < 0:
        raise ValueError("--max-improvement-std must be non-negative")
    if args.max_teacher_relative_degradation < 0:
        raise ValueError("--max-teacher-relative-degradation must be non-negative")
    if args.max_formula_relative_degradation < 0:
        raise ValueError("--max-formula-relative-degradation must be non-negative")

    official_path = Path(args.official_fold_csv)
    kan_output_dir = Path(args.kan_output_dir)
    kan_path = (
        Path(args.kan_fold_csv)
        if args.kan_fold_csv
        else latest_kan_csv(kan_output_dir, args.dataset)
    )
    official_rows = read_csv(official_path)
    kan_rows = read_csv(kan_path)
    if not official_rows or not kan_rows:
        raise ValueError("Official and KAN fold CSV files must both contain rows")

    task_type = str(
        next(
            (
                row.get("task_type")
                for row in kan_rows
                if row.get("model") == args.model and row.get("task_type")
            ),
            official_rows[0].get("task_type", "regression"),
        )
    )
    classification = task_type == "classification"
    formula_targets = {"formula", "spline-symbolic", "symbolic-kan"}
    is_formula_target = args.comparison_target in formula_targets
    if classification and is_formula_target:
        raise ValueError("Symbolic formula comparison currently supports regression only")

    official_metric = "rocauc" if classification else "mae"
    kan_metric = {
        "formula": "simple_formula_test_target_mae",
        "spline-symbolic": "spline_symbolic_test_target_mae",
        "symbolic-kan": "symbolic_kan_hard_test_mae",
    }.get(
        args.comparison_target,
        "test_rocauc" if classification else "test_mae",
    )
    fidelity_field = {
        "formula": "simple_formula_test_fidelity_r2_pct",
        "spline-symbolic": "spline_symbolic_test_fidelity_r2_pct",
        "symbolic-kan": "symbolic_kan_test_fidelity_r2_pct",
    }.get(args.comparison_target)
    inputs_field = {
        "formula": "simple_formula_inputs",
        "spline-symbolic": "spline_symbolic_active_features",
        "symbolic-kan": "symbolic_kan_active_features",
    }.get(args.comparison_target)
    terms_field = {
        "formula": "simple_formula_n_terms",
        "spline-symbolic": "spline_symbolic_n_edges",
        "symbolic-kan": "symbolic_kan_active_units",
    }.get(args.comparison_target)
    official_by_fold = rows_by_fold(
        official_rows, official_metric, "official MODNet"
    )
    kan_by_fold = rows_by_fold(
        kan_rows, kan_metric, args.comparison_target, model=args.model
    )
    teacher_by_fold = rows_by_fold(
        kan_rows,
        "test_rocauc" if classification else "test_mae",
        "KAN teacher",
        model=args.model,
    )
    expected = set(range(args.expected_folds))
    if set(official_by_fold) != expected:
        raise ValueError(
            f"Official folds are {sorted(official_by_fold)}, expected {sorted(expected)}"
        )
    if set(kan_by_fold) != expected:
        raise ValueError(
            f"KAN folds are {sorted(kan_by_fold)}, expected {sorted(expected)}"
        )
    if set(teacher_by_fold) != expected:
        raise ValueError(
            f"KAN teacher folds are {sorted(teacher_by_fold)}, expected {sorted(expected)}"
        )

    fold_rows: list[dict[str, Any]] = []
    improvements = []
    teacher_improvements = []
    wins = 0
    teacher_wins = 0
    for fold in sorted(expected):
        official_value = finite_float(
            official_by_fold[fold][official_metric],
            f"official fold {fold} {official_metric}",
        )
        kan_value = finite_float(
            kan_by_fold[fold][kan_metric],
            f"{args.comparison_target} fold {fold} {kan_metric}",
        )
        teacher_metric = "test_rocauc" if classification else "test_mae"
        teacher_value = finite_float(
            teacher_by_fold[fold][teacher_metric],
            f"KAN teacher fold {fold} {teacher_metric}",
        )
        if classification:
            improvement = kan_value - official_value
            better = kan_value > official_value
            teacher_improvement = teacher_value - official_value
            teacher_better = teacher_value > official_value
        else:
            if official_value == 0:
                raise ValueError(f"Official MAE is zero for fold {fold}")
            improvement = (official_value - kan_value) / abs(official_value)
            better = kan_value < official_value
            teacher_improvement = (
                official_value - teacher_value
            ) / abs(official_value)
            teacher_better = teacher_value < official_value
        improvements.append(improvement)
        teacher_improvements.append(teacher_improvement)
        wins += int(better)
        teacher_wins += int(teacher_better)
        fold_rows.append(
            {
                "dataset": args.dataset,
                "fold": fold,
                "task_type": task_type,
                "official_metric": official_metric,
                "official_value": official_value,
                "kan_model": args.model,
                "teacher_metric": teacher_metric,
                "teacher_value": teacher_value,
                "teacher_improvement": teacher_improvement,
                "teacher_improvement_pct": 100.0 * teacher_improvement,
                "teacher_gap_vs_official_pct": -100.0 * teacher_improvement,
                "teacher_better": teacher_better,
                "comparison_target": args.comparison_target,
                "candidate_metric": kan_metric,
                "candidate_value": kan_value,
                "improvement": improvement,
                "improvement_pct": 100.0 * improvement,
                "candidate_gap_vs_official_pct": -100.0 * improvement,
                "candidate_better": better,
                "formula_teacher_fidelity_r2": (
                    finite_float(
                        kan_by_fold[fold].get(fidelity_field),
                        f"formula fold {fold} fidelity R2 percent",
                    )
                    / 100.0
                    if is_formula_target and fidelity_field is not None
                    else None
                ),
                "formula_inputs": (
                    kan_by_fold[fold].get(inputs_field, "")
                    if is_formula_target and inputs_field is not None
                    else ""
                ),
                "formula_terms": (
                    kan_by_fold[fold].get(terms_field, "")
                    if is_formula_target and terms_field is not None
                    else ""
                ),
            }
        )

    official_mean = mean(row["official_value"] for row in fold_rows)
    candidate_mean = mean(row["candidate_value"] for row in fold_rows)
    teacher_mean = mean(row["teacher_value"] for row in fold_rows)
    mean_improvement = mean(improvements)
    improvement_std = pstdev(improvements)
    teacher_mean_improvement = mean(teacher_improvements)
    teacher_improvement_std = pstdev(teacher_improvements)
    required_improvement = (
        args.min_absolute_improvement
        if classification
        else args.min_relative_improvement
    )
    performance_passes = (
        mean_improvement >= required_improvement and wins >= args.min_fold_wins
    )
    teacher_closeness_passes = (
        teacher_mean_improvement >= -args.max_teacher_relative_degradation
    )
    formula_closeness_passes = (
        mean_improvement >= -args.max_formula_relative_degradation
        if is_formula_target
        else True
    )

    formulas: list[dict[str, Any]] = []
    stability: dict[str, Any] | None = None
    stability_passes = True
    if is_formula_target:
        formulas, feature_sets = load_fold_formulas(
            kan_by_fold,
            kan_output_dir,
            comparison_target=args.comparison_target,
        )
        stability = formula_stability(
            formulas,
            feature_sets,
            expected_folds=args.expected_folds,
        )
        mean_fidelity = stability["mean_formula_teacher_fidelity_r2"]
        stability_passes = (
            mean_fidelity is not None
            and mean_fidelity >= args.min_formula_fidelity_r2
            and stability["mean_pairwise_feature_jaccard"]
            >= args.min_feature_jaccard
            and stability["mean_pairwise_operator_jaccard"]
            >= args.min_operator_jaccard
            and improvement_std <= args.max_improvement_std
        )

    params = [
        finite_float(row.get("params_after_prune"), "KAN params_after_prune")
        for row in kan_rows
        if row.get("model") == args.model and row.get("params_after_prune")
    ]
    closeness_passes = teacher_closeness_passes and formula_closeness_passes
    passes = closeness_passes and stability_passes
    summary = {
        "dataset": args.dataset,
        "task_type": task_type,
        "official_fold_csv": str(official_path),
        "kan_fold_csv": str(kan_path),
        "kan_model": args.model,
        "comparison_target": args.comparison_target,
        "folds": args.expected_folds,
        "official_metric": official_metric,
        "official_mean": official_mean,
        "teacher_metric": "test_rocauc" if classification else "test_mae",
        "teacher_mean": teacher_mean,
        "kan_mean": teacher_mean,
        "teacher_mean_improvement": teacher_mean_improvement,
        "teacher_mean_improvement_pct": 100.0 * teacher_mean_improvement,
        "teacher_mean_degradation_pct": -100.0 * teacher_mean_improvement,
        "kan_gap_vs_official_pct": -100.0 * teacher_mean_improvement,
        "teacher_improvement_std": teacher_improvement_std,
        "teacher_improvement_std_pct": 100.0 * teacher_improvement_std,
        "teacher_fold_wins": teacher_wins,
        "candidate_metric": kan_metric,
        "candidate_mean": candidate_mean,
        "formula_mean": (
            candidate_mean if is_formula_target else None
        ),
        "spline_symbolic_mean": (
            candidate_mean if args.comparison_target == "spline-symbolic" else None
        ),
        "symbolic_kan_hard_mean": (
            candidate_mean if args.comparison_target == "symbolic-kan" else None
        ),
        "mean_improvement": mean_improvement,
        "mean_improvement_pct": 100.0 * mean_improvement,
        "formula_gap_vs_official_pct": (
            -100.0 * mean_improvement
            if is_formula_target
            else None
        ),
        "improvement_std": improvement_std,
        "improvement_std_pct": 100.0 * improvement_std,
        "required_improvement": required_improvement,
        "required_improvement_pct": 100.0 * required_improvement,
        "candidate_fold_wins": wins,
        "formula_fold_wins": (
            wins if is_formula_target else None
        ),
        "required_fold_wins": args.min_fold_wins,
        "kan_params_mean": mean(params) if params else None,
        "formula_stability": stability,
        "required_mean_formula_fidelity_r2": args.min_formula_fidelity_r2,
        "required_mean_feature_jaccard": args.min_feature_jaccard,
        "required_mean_operator_jaccard": args.min_operator_jaccard,
        "maximum_improvement_std": args.max_improvement_std,
        "maximum_teacher_relative_degradation": args.max_teacher_relative_degradation,
        "maximum_formula_relative_degradation": args.max_formula_relative_degradation,
        "passes_superiority_gate": performance_passes,
        "passes_teacher_closeness_gate": teacher_closeness_passes,
        "passes_formula_closeness_gate": formula_closeness_passes,
        "passes_closeness_gate": closeness_passes,
        "passes_stability_gate": stability_passes,
        "passes_fixed5fold_gate": passes,
        "fold_formulas": formulas,
        "comparison_note": (
            "Completed official MODNet result versus outer-test predictions from "
            f"the fixed {args.comparison_target}; no outer-fold hyperparameter selection."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"fixed5fold-comparison-{args.dataset}.csv"
    json_path = output_dir / f"fixed5fold-comparison-{args.dataset}.json"
    write_csv(csv_path, fold_rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path: Path | None = None
    if stability is not None:
        report_path = output_dir / f"symbolic-formulas-5fold-{args.dataset}.txt"
        report_path.write_text(
            "\n".join(
                formula_report_lines(args.dataset, formulas, stability)
            ),
            encoding="utf-8",
        )

    status = "PASS" if passes else "FAIL"
    unit = "ROC-AUC points" if classification else "relative MAE"
    print(
        f"{args.dataset}: {status} | official={official_mean:.6g} | "
        f"KAN teacher={teacher_mean:.6g} ({teacher_mean_improvement:+.3%}) | "
        f"{args.comparison_target}={candidate_mean:.6g} ({mean_improvement:+.3%}) | "
        f"teacher wins={teacher_wins}/{args.expected_folds} | "
        f"{args.comparison_target} wins={wins}/{args.expected_folds} | "
        f"{unit}",
        flush=True,
    )
    if stability is not None:
        print(
            "Formula stability: "
            f"mean fidelity R2={stability['mean_formula_teacher_fidelity_r2']:.6g}, "
            f"mean feature Jaccard={stability['mean_pairwise_feature_jaccard']:.6g}, "
            f"mean operator Jaccard={stability['mean_pairwise_operator_jaccard']:.6g}",
            flush=True,
        )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    if report_path is not None:
        print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
