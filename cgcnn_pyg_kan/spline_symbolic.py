from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .kan import KANLinear


SPLINE_SYMBOLIC_FUNCTIONS = (
    "identity",
    "square",
    "cube",
    "sin",
    "cos",
    "tan",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "reciprocal",
    "abs",
    "arctan",
    "gaussian",
)

FUNCTION_COMPLEXITY = {
    "identity": 1,
    "square": 2,
    "cube": 3,
    "sin": 2,
    "cos": 2,
    "tan": 3,
    "tanh": 3,
    "exp": 2,
    "log": 2,
    "sqrt": 2,
    "reciprocal": 2,
    "abs": 3,
    "arctan": 4,
    "gaussian": 3,
}


def _function(name: str, values: np.ndarray, epsilon: float) -> np.ndarray:
    if name == "identity":
        result = values
    elif name == "square":
        result = values**2
    elif name == "cube":
        result = values**3
    elif name == "sin":
        result = np.sin(values)
    elif name == "cos":
        result = np.cos(values)
    elif name == "tan":
        result = np.clip(np.tan(values), -1.0 / epsilon, 1.0 / epsilon)
    elif name == "tanh":
        result = np.tanh(values)
    elif name == "exp":
        result = np.exp(np.clip(values, -8.0, 8.0))
    elif name == "log":
        result = np.log(np.abs(values) + epsilon)
    elif name == "sqrt":
        result = np.sign(values) * np.sqrt(np.abs(values))
    elif name == "reciprocal":
        result = values / (values**2 + epsilon**2)
    elif name == "abs":
        result = np.abs(values)
    elif name == "arctan":
        result = np.arctan(values)
    elif name == "gaussian":
        result = np.exp(-np.clip(values**2, 0.0, 20.0))
    else:
        raise ValueError(f"Unsupported spline symbolic function {name!r}")
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0 / epsilon,
        neginf=-1.0 / epsilon,
    )


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    coefficient = (
        float(centered @ (y - y.mean()) / denominator)
        if denominator > 1e-14
        else 0.0
    )
    intercept = float(y.mean() - coefficient * x.mean())
    prediction = coefficient * x + intercept
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    return coefficient, intercept, 1.0 - residual / max(total, 1e-14)


def fit_edge_function(
    x: np.ndarray,
    y: np.ndarray,
    functions: Sequence[str],
    *,
    search_range: float,
    grid_size: int,
    iterations: int,
    complexity_weight: float,
    epsilon: float,
) -> dict[str, float | int | str]:
    """Pykan-style fit of ``y = c*f(a*x+b)+d`` and symbolic suggestion."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    candidates: list[dict[str, float | int | str]] = []
    for name in functions:
        if name == "identity":
            c, d, r2 = _linear_fit(x, y)
            candidate = {
                "function": name,
                "a": 1.0,
                "b": 0.0,
                "c": c,
                "d": d,
                "r2": r2,
                "complexity": FUNCTION_COMPLEXITY[name],
            }
            candidates.append(candidate)
            continue

        a_low, a_high = -search_range, search_range
        b_low, b_high = -search_range, search_range
        best_a = best_b = 0.0
        for _ in range(iterations):
            a_values = np.linspace(a_low, a_high, grid_size)
            b_values = np.linspace(b_low, b_high, grid_size)
            best_score = -np.inf
            for a in a_values:
                library = _function(
                    name,
                    a * x[:, None] + b_values[None, :],
                    epsilon,
                )
                library -= library.mean(axis=0, keepdims=True)
                target = y - y.mean()
                numerator = np.sum(library * target[:, None], axis=0) ** 2
                denominator = (
                    np.sum(library**2, axis=0)
                    * max(float(np.sum(target**2)), 1e-14)
                    + 1e-14
                )
                scores = numerator / denominator
                index = int(np.nanargmax(scores))
                if float(scores[index]) > best_score:
                    best_score = float(scores[index])
                    best_a = float(a)
                    best_b = float(b_values[index])
            a_step = (a_high - a_low) / max(grid_size - 1, 1)
            b_step = (b_high - b_low) / max(grid_size - 1, 1)
            a_low, a_high = best_a - a_step, best_a + a_step
            b_low, b_high = best_b - b_step, best_b + b_step
        basis = _function(name, best_a * x + best_b, epsilon)
        c, d, r2 = _linear_fit(basis, y)
        candidates.append(
            {
                "function": name,
                "a": best_a,
                "b": best_b,
                "c": c,
                "d": d,
                "r2": r2,
                "complexity": FUNCTION_COMPLEXITY[name],
            }
        )

    for candidate in candidates:
        r2 = min(float(candidate["r2"]), 1.0)
        r2_loss = float(np.log2(max(1e-12, 1.0 + 1e-5 - r2)))
        candidate["selection_loss"] = (
            complexity_weight * float(candidate["complexity"])
            + (1.0 - complexity_weight) * r2_loss
        )
    return min(candidates, key=lambda item: float(item["selection_loss"]))


@torch.no_grad()
def _all_edge_values(layer: KANLinear, values: np.ndarray) -> np.ndarray:
    inputs = torch.as_tensor(
        values,
        dtype=layer.base_weight.dtype,
        device=layer.base_weight.device,
    )
    base = F.silu(inputs)[:, None, :] * layer.base_weight.detach()[None, :, :]
    spline = torch.einsum(
        "nib,oib->noi",
        layer.b_splines(inputs),
        layer.scaled_spline_weight.detach(),
    )
    return (base + spline).cpu().numpy()


def _evaluate(spec: dict[str, Any], values: np.ndarray, epsilon: float) -> np.ndarray:
    basis = _function(
        str(spec["function"]),
        float(spec["a"]) * values + float(spec["b"]),
        epsilon,
    )
    return float(spec["c"]) * basis + float(spec["d"])


def _expression(spec: dict[str, Any], variable: str) -> str:
    argument = (
        f"{float(spec['a']):.8g}*{variable}"
        f"{float(spec['b']):+.8g}"
    )
    function = str(spec["function"])
    body = {
        "identity": f"({argument})",
        "square": f"({argument})^2",
        "cube": f"({argument})^3",
        "sin": f"sin({argument})",
        "cos": f"cos({argument})",
        "tan": f"protected_tan({argument})",
        "tanh": f"tanh({argument})",
        "exp": f"protected_exp({argument})",
        "log": f"log(abs({argument})+eps)",
        "sqrt": f"signed_sqrt({argument})",
        "reciprocal": f"protected_reciprocal({argument})",
        "abs": f"abs({argument})",
        "arctan": f"arctan({argument})",
        "gaussian": f"exp(-({argument})^2)",
    }[function]
    return (
        f"{float(spec['c']):.8g}*({body})"
        f"{float(spec['d']):+.8g}"
    )


def symbolify_spline_kan(
    layers: Sequence[KANLinear],
    x_fit: np.ndarray,
    x_test: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    *,
    functions: Sequence[str] = SPLINE_SYMBOLIC_FUNCTIONS,
    input_edges_per_hidden: int = 5,
    output_edges_per_target: int = 4,
    max_fit_samples: int = 1024,
    search_range: float = 10.0,
    grid_size: int = 21,
    iterations: int = 2,
    complexity_weight: float = 0.2,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any], str]:
    if len(layers) != 2 or not all(isinstance(layer, KANLinear) for layer in layers):
        raise ValueError(
            "Spline auto-symbolic requires exactly two KANLinear layers"
        )
    first, second = layers
    if first.out_features != second.in_features:
        raise ValueError("Spline KAN layers are not composable")
    if len(x_fit) > max_fit_samples:
        indices = np.linspace(0, len(x_fit) - 1, max_fit_samples, dtype=int)
        x_sample = np.asarray(x_fit)[indices]
    else:
        x_sample = np.asarray(x_fit)

    first_numeric = _all_edge_values(first, x_sample)
    hidden_numeric = first_numeric.sum(axis=-1)
    second_numeric = _all_edge_values(second, hidden_numeric)
    output_scores = np.sqrt(np.mean(second_numeric**2, axis=0))
    input_scores = np.sqrt(np.mean(first_numeric**2, axis=0))

    selected_output: dict[int, list[int]] = {}
    active_hidden: set[int] = set()
    for target_index in range(second.out_features):
        selected = np.argsort(output_scores[target_index])[
            -min(output_edges_per_target, second.in_features) :
        ][::-1]
        selected_output[target_index] = [int(value) for value in selected]
        active_hidden.update(selected_output[target_index])
    selected_input = {
        hidden: [
            int(value)
            for value in np.argsort(input_scores[hidden])[
                -min(input_edges_per_hidden, first.in_features) :
            ][::-1]
        ]
        for hidden in sorted(active_hidden)
    }

    first_specs: dict[int, list[dict[str, Any]]] = {}
    for hidden, inputs in selected_input.items():
        first_specs[hidden] = []
        for input_index in inputs:
            spec = fit_edge_function(
                x_sample[:, input_index],
                first_numeric[:, hidden, input_index],
                functions,
                search_range=search_range,
                grid_size=grid_size,
                iterations=iterations,
                complexity_weight=complexity_weight,
                epsilon=epsilon,
            )
            first_specs[hidden].append(
                {
                    **spec,
                    "layer": 0,
                    "input_index": input_index,
                    "output_index": hidden,
                }
            )

    second_specs: dict[int, list[dict[str, Any]]] = {}
    for target_index, hidden_indices in selected_output.items():
        second_specs[target_index] = []
        for hidden in hidden_indices:
            spec = fit_edge_function(
                hidden_numeric[:, hidden],
                second_numeric[:, target_index, hidden],
                functions,
                search_range=search_range,
                grid_size=grid_size,
                iterations=iterations,
                complexity_weight=complexity_weight,
                epsilon=epsilon,
            )
            second_specs[target_index].append(
                {
                    **spec,
                    "layer": 1,
                    "input_index": hidden,
                    "output_index": target_index,
                }
            )

    x_test = np.asarray(x_test, dtype=np.float64)
    symbolic_hidden = np.zeros((len(x_test), first.out_features), dtype=np.float64)
    for hidden, records in first_specs.items():
        for record in records:
            symbolic_hidden[:, hidden] += _evaluate(
                record,
                x_test[:, int(record["input_index"])],
                epsilon,
            )
    symbolic_output = np.zeros((len(x_test), second.out_features), dtype=np.float64)
    for target_index, records in second_specs.items():
        for record in records:
            symbolic_output[:, target_index] += _evaluate(
                record,
                symbolic_hidden[:, int(record["input_index"])],
                epsilon,
            )

    payload: dict[str, Any] = {
        "method": "spline_kan_edge_auto_symbolic",
        "reference_algorithm": "pykan fit_params/suggest_symbolic/auto_symbolic",
        "architecture": [first.in_features, first.out_features, second.out_features],
        "function_library": list(functions),
        "input_edges_per_hidden": input_edges_per_hidden,
        "output_edges_per_target": output_edges_per_target,
        "fit_samples": len(x_sample),
        "targets": [],
    }
    report = [
        "Spline KAN discrete formulas",
        "method = pykan-style edge-wise auto_symbolic",
        "",
    ]
    for target_index, target_name in enumerate(target_names):
        hidden_indices = selected_output[target_index]
        active_inputs = sorted(
            {
                int(record["input_index"])
                for hidden in hidden_indices
                for record in first_specs[hidden]
            }
        )
        hidden_definitions = []
        edge_records: list[dict[str, Any]] = []
        for hidden in hidden_indices:
            expression = " + ".join(
                _expression(record, f"x{record['input_index']}")
                for record in first_specs[hidden]
            )
            hidden_definitions.append(
                {"variable": f"h{hidden}", "expression": expression}
            )
            edge_records.extend(first_specs[hidden])
        expression = " + ".join(
            _expression(record, f"h{record['input_index']}")
            for record in second_specs[target_index]
        )
        edge_records.extend(second_specs[target_index])
        operators = sorted(
            {str(record["function"]) for record in edge_records}
        )
        record = {
            "target": str(target_name),
            "expression": expression,
            "hidden_definitions": hidden_definitions,
            "active_feature_names": [
                str(feature_names[index]) for index in active_inputs
            ],
            "variable_definitions": [
                {
                    "variable": f"x{index}",
                    "feature": str(feature_names[index]),
                    "expression": f"preprocessed({feature_names[index]!r})",
                }
                for index in active_inputs
            ],
            "operators": operators,
            "term_names": operators,
            "edges": edge_records,
            "n_edges": len(edge_records),
            "mean_edge_fit_r2": float(
                np.mean([float(item["r2"]) for item in edge_records])
            ),
            "min_edge_fit_r2": float(
                np.min([float(item["r2"]) for item in edge_records])
            ),
        }
        payload["targets"].append(record)
        report.append(f"target = {target_name}")
        for definition in record["variable_definitions"]:
            report.append(
                f"  {definition['variable']} = {definition['expression']}"
            )
        for definition in hidden_definitions:
            report.append(
                f"  {definition['variable']} = {definition['expression']}"
            )
        report.append(f"  y = {expression}")
        report.append("")
    operator_counts = Counter(
        operator
        for target in payload["targets"]
        for operator in target["operators"]
    )
    payload["operator_target_counts"] = dict(sorted(operator_counts.items()))
    return symbolic_output.astype(np.float32), payload, "\n".join(report)
