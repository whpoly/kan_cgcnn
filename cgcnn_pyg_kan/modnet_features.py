from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FeaturePreset = Literal[
    "pymatgen-composition",
    "pymatgen-structure",
    "matminer-composition",
    "matminer-structure-lite",
]


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_columns: list[str]
    relevance: dict[str, float]


def make_feature_frame(
    materials: Sequence,
    preset: FeaturePreset = "pymatgen-composition",
    n_jobs: int = 1,
) -> pd.DataFrame:
    if preset == "pymatgen-composition":
        return _pymatgen_feature_frame(materials, include_structure=False)
    if preset == "pymatgen-structure":
        return _pymatgen_feature_frame(materials, include_structure=True)
    if preset == "matminer-composition":
        return _matminer_composition_frame(materials, n_jobs=n_jobs)
    if preset == "matminer-structure-lite":
        return _matminer_structure_lite_frame(materials, n_jobs=n_jobs)
    raise ValueError(f"unsupported feature preset {preset!r}")


def numeric_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.drop(columns=[col for col in frame.columns if col.startswith("_")], errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, ~numeric.columns.duplicated()]
    keep = []
    for column in numeric.columns:
        series = numeric[column]
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        keep.append(column)
    if not keep:
        raise ValueError("no usable numeric feature columns were found")
    return numeric[keep]


def select_relevance_redundancy_features(
    frame: pd.DataFrame,
    targets: Sequence[float],
    n_features: int,
    random_state: int = 0,
    task_type: Literal["regression", "classification"] = "regression",
) -> FeatureSelectionResult:
    numeric = numeric_feature_frame(frame)
    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(y) != len(numeric):
        raise ValueError("targets must have the same number of rows as frame")

    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    scaled = MinMaxScaler(feature_range=(-0.5, 0.5)).fit_transform(imputed)
    relevance = _feature_relevance(
        scaled,
        y,
        random_state=random_state,
        task_type=task_type,
    )

    n_select = min(n_features, scaled.shape[1])
    selected = _rr_greedy_select(scaled, relevance, n_select)
    columns = list(numeric.columns)
    selected_columns = [columns[idx] for idx in selected]
    relevance_map = {columns[idx]: float(relevance[idx]) for idx in range(len(columns))}
    return FeatureSelectionResult(selected_columns=selected_columns, relevance=relevance_map)


class MODNetFeatureProcessor:
    def __init__(
        self,
        n_features: int = 256,
        scaler: Literal["minmax", "standard", "none"] = "minmax",
        impute_strategy: str = "median",
        random_state: int = 0,
        task_type: Literal["regression", "classification"] = "regression",
    ) -> None:
        self.n_features = n_features
        self.scaler = scaler
        self.impute_strategy = impute_strategy
        self.random_state = random_state
        self.task_type = task_type
        self.numeric_columns_: list[str] | None = None
        self.selected_columns_: list[str] | None = None
        self.relevance_: dict[str, float] | None = None
        self.pipeline_: Pipeline | None = None

    def fit(self, frame: pd.DataFrame, targets: Sequence[float]) -> MODNetFeatureProcessor:
        numeric = numeric_feature_frame(frame)
        self.numeric_columns_ = list(numeric.columns)
        selection = select_relevance_redundancy_features(
            numeric,
            targets,
            n_features=self.n_features,
            random_state=self.random_state,
            task_type=self.task_type,
        )
        self.selected_columns_ = selection.selected_columns
        self.relevance_ = selection.relevance

        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy=self.impute_strategy))]
        if self.scaler == "minmax":
            steps.append(("scaler", MinMaxScaler(feature_range=(-0.5, 0.5))))
        elif self.scaler == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.scaler != "none":
            raise ValueError("scaler must be 'minmax', 'standard', or 'none'")

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(numeric[self.selected_columns_])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.numeric_columns_ is None or self.selected_columns_ is None or self.pipeline_ is None:
            raise RuntimeError("processor has not been fit")
        numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric = numeric.reindex(columns=self.selected_columns_)
        return self.pipeline_.transform(numeric).astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame, targets: Sequence[float]) -> np.ndarray:
        return self.fit(frame, targets).transform(frame)


def _feature_relevance(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    task_type: Literal["regression", "classification"],
) -> np.ndarray:
    if task_type == "classification":
        y_labels = np.asarray(y).astype(int)
        if len(np.unique(y_labels)) < 2:
            return _absolute_correlations(x, y_labels)
        relevance = mutual_info_classif(
            x,
            y_labels,
            random_state=random_state,
            discrete_features=False,
        )
        return np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)

    if len(y) < 4 or np.nanstd(y) <= 1e-12:
        relevance = _absolute_correlations(x, y)
    else:
        n_neighbors = max(1, min(3, len(y) - 1))
        relevance = mutual_info_regression(
            x,
            y,
            random_state=random_state,
            n_neighbors=n_neighbors,
        )
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(relevance <= 0):
        relevance = _absolute_correlations(x, y)
    return np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)


def _absolute_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = x - np.nanmean(x, axis=0, keepdims=True)
    centered_y = y - np.nanmean(y)
    numerator = np.nansum(centered_x * centered_y[:, None], axis=0)
    denominator = np.sqrt(
        np.nansum(centered_x**2, axis=0) * np.nansum(centered_y**2)
    )
    denominator = np.maximum(denominator, 1e-12)
    return np.abs(numerator / denominator)


def _rr_greedy_select(
    x: np.ndarray,
    relevance: np.ndarray,
    n_features: int,
) -> list[int]:
    if n_features >= x.shape[1]:
        return list(np.argsort(-relevance))

    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    selected = [int(np.argmax(relevance))]
    remaining = set(range(x.shape[1]))
    remaining.remove(selected[0])

    while len(selected) < n_features and remaining:
        rem = np.array(sorted(remaining), dtype=int)
        redundancy = corr[np.ix_(rem, selected)].max(axis=1)
        n_chosen = len(selected)
        power = max(0.1, 4.5 - 0.4 * (n_chosen**0.4))
        constant = min(1e5, 1e-6 * (n_chosen**3))
        scores = relevance[rem] / (np.power(redundancy, power) + constant)
        best = int(rem[int(np.argmax(scores))])
        selected.append(best)
        remaining.remove(best)
    return selected


def _pymatgen_feature_frame(
    materials: Sequence,
    include_structure: bool,
) -> pd.DataFrame:
    rows = []
    for material in materials:
        composition = _as_composition(material)
        row = _composition_stats(composition)
        if include_structure:
            row.update(_structure_stats(material))
        rows.append(row)
    return pd.DataFrame(rows)


def _composition_stats(composition: Composition) -> dict[str, float]:
    el_amt = composition.element_composition.as_dict()
    total = float(sum(el_amt.values()))
    elements = list(composition.elements)
    fractions = np.array([el_amt[element.symbol] / total for element in elements], dtype=float)

    row: dict[str, float] = {
        "n_elements": float(len(elements)),
        "total_atoms_reduced": float(sum(composition.reduced_composition.as_dict().values())),
    }
    for atomic_number in range(1, 119):
        row[f"frac_Z_{atomic_number}"] = 0.0
    for element, fraction in zip(elements, fractions):
        row[f"frac_Z_{int(element.Z)}"] = float(fraction)

    properties = {
        "Z": [float(element.Z) for element in elements],
        "group": [_safe_float(getattr(element, "group", None)) for element in elements],
        "row": [_safe_float(getattr(element, "row", None)) for element in elements],
        "atomic_mass": [_safe_float(getattr(element, "atomic_mass", None)) for element in elements],
        "electronegativity": [_safe_float(getattr(element, "X", None)) for element in elements],
        "atomic_radius": [_safe_float(getattr(element, "atomic_radius", None)) for element in elements],
        "mendeleev_number": [_safe_float(getattr(element, "mendeleev_no", None)) for element in elements],
    }
    for name, values in properties.items():
        values_arr = np.asarray(values, dtype=float)
        mean = float(np.dot(fractions, values_arr))
        variance = float(np.dot(fractions, (values_arr - mean) ** 2))
        row[f"{name}_mean"] = mean
        row[f"{name}_std"] = variance**0.5
        row[f"{name}_min"] = float(np.min(values_arr))
        row[f"{name}_max"] = float(np.max(values_arr))
        row[f"{name}_range"] = float(np.max(values_arr) - np.min(values_arr))
    return row


def _structure_stats(material) -> dict[str, float]:
    if not hasattr(material, "volume"):
        return {}
    n_sites = max(1, len(material))
    return {
        "n_sites": float(n_sites),
        "volume": _safe_float(getattr(material, "volume", None)),
        "volume_per_atom": _safe_float(getattr(material, "volume", None)) / n_sites,
        "density": _safe_float(getattr(material, "density", None)),
    }


def _matminer_composition_frame(materials: Sequence, n_jobs: int) -> pd.DataFrame:
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

    featurizer = MultipleFeaturizer(
        [
            ElementProperty.from_preset("magpie"),
            Stoichiometry(),
            ValenceOrbital(),
        ]
    )
    featurizer.set_n_jobs(n_jobs)
    frame = pd.DataFrame({"composition": [_as_composition(material) for material in materials]})
    return _matminer_numeric_frame(
        featurizer.featurize_dataframe(
            frame,
            "composition",
            ignore_errors=True,
            inplace=False,
        )
    )


def _matminer_structure_lite_frame(materials: Sequence, n_jobs: int) -> pd.DataFrame:
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
    from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
    from matminer.featurizers.structure.misc import StructureComposition

    featurizer = MultipleFeaturizer(
        [
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(Stoichiometry()),
            StructureComposition(ValenceOrbital()),
            DensityFeatures(),
            GlobalSymmetryFeatures(),
        ]
    )
    featurizer.set_n_jobs(n_jobs)
    frame = pd.DataFrame({"structure": list(materials)})
    return _matminer_numeric_frame(
        featurizer.featurize_dataframe(
            frame,
            "structure",
            ignore_errors=True,
            inplace=False,
        )
    )


def _matminer_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [column for column in ("composition", "structure") if column in frame.columns]
    return frame.drop(columns=drop_cols, errors="ignore")


def _as_composition(material) -> Composition:
    if isinstance(material, Composition):
        return material
    composition = getattr(material, "composition", None)
    if composition is not None:
        return composition
    return Composition(str(material))


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return number
