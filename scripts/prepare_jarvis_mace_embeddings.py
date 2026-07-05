from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a JARVIS dataset and cache frozen MACE crystal descriptors "
            "for downstream Linear/MLP/KAN head benchmarks."
        )
    )
    parser.add_argument("--dataset", default="dft_3d")
    parser.add_argument("--target", default="optb88vdw_bandgap")
    parser.add_argument(
        "--mace-model",
        default="small",
        choices=["small", "medium", "large"],
        help="MACE-MP checkpoint size used as the frozen structure encoder.",
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--pooling", nargs="+", default=["mean", "std"], choices=["mean", "std", "max", "min"])
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--equivariants", dest="invariants_only", action="store_false")
    parser.set_defaults(invariants_only=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=0,
        help="Skip structures larger than this. Use 0 to disable the cap.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed structure.")
    return parser.parse_args()


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def default_output_path(args: argparse.Namespace) -> Path:
    sample_part = f"n{args.max_samples}" if args.max_samples else "all"
    atom_part = f"maxatoms{args.max_atoms}" if args.max_atoms else "allatoms"
    pool_part = "-".join(args.pooling)
    name = (
        f"{sanitize(args.dataset)}_{sanitize(args.target)}_"
        f"mace-{sanitize(args.mace_model)}_{pool_part}_{sample_part}_{atom_part}.npz"
    )
    return ROOT / "data" / "jarvis_mace" / name


def value_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "na", "nan", "none", "null", "--"}:
            return None
        value = cleaned
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_jarvis_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from jarvis.db.figshare import data
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "jarvis-tools is required. Install it with: pip install -r requirements-jarvis-mace.txt"
        ) from exc

    entries = data(args.dataset)
    filtered: list[dict[str, Any]] = []
    skipped_no_target = 0
    skipped_no_atoms = 0
    skipped_too_large = 0
    max_atoms = args.max_atoms if args.max_atoms and args.max_atoms > 0 else None

    for index, entry in enumerate(entries):
        target = value_to_float(entry.get(args.target))
        if target is None:
            skipped_no_target += 1
            continue
        atom_dict = entry.get("atoms")
        if not atom_dict:
            skipped_no_atoms += 1
            continue
        if max_atoms is not None and len(atom_dict.get("elements", [])) > max_atoms:
            skipped_too_large += 1
            continue
        item = dict(entry)
        item["_target_float"] = target
        item["_source_index"] = index
        filtered.append(item)

    if args.max_samples and args.max_samples < len(filtered):
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(filtered), size=args.max_samples, replace=False))
        filtered = [filtered[int(idx)] for idx in selected]

    print(
        f"Loaded {len(entries)} {args.dataset} rows; kept {len(filtered)} with finite "
        f"{args.target}. Skipped target={skipped_no_target}, atoms={skipped_no_atoms}, "
        f"too_large={skipped_too_large}.",
        flush=True,
    )
    return filtered


def build_mace_calculator(args: argparse.Namespace):
    try:
        from mace.calculators import mace_mp
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "mace-torch is required. Install it with: pip install -r requirements-jarvis-mace.txt"
        ) from exc

    kwargs = {
        "model": args.mace_model,
        "device": args.device,
        "default_dtype": args.default_dtype,
    }
    try:
        return mace_mp(**kwargs)
    except TypeError:
        kwargs.pop("default_dtype", None)
        return mace_mp(**kwargs)


def jarvis_entry_to_ase(entry: dict[str, Any]):
    try:
        from jarvis.core.atoms import Atoms as JarvisAtoms
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("jarvis-tools is required to convert JARVIS atoms.") from exc

    atoms = JarvisAtoms.from_dict(entry["atoms"]).ase_converter()
    atoms.pbc = True
    return atoms


def reduced_formula(formula: str) -> str:
    try:
        from pymatgen.core import Composition

        return Composition(formula).reduced_formula
    except Exception:
        return formula


def descriptor_to_numpy(raw: Any) -> np.ndarray:
    if isinstance(raw, dict):
        for key in ("descriptors", "descriptor", "node_feats", "features"):
            if key in raw:
                raw = raw[key]
                break
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    array = np.asarray(raw, dtype=np.float32)
    if array.size == 0:
        raise ValueError("MACE returned an empty descriptor array")
    return np.squeeze(array)


def normalize_descriptor_shape(descriptor: np.ndarray, num_atoms: int) -> np.ndarray:
    if descriptor.ndim == 1:
        return descriptor.reshape(1, -1)
    if descriptor.ndim == 2:
        return descriptor
    if descriptor.ndim == 3:
        if descriptor.shape[0] == 1:
            return descriptor[0]
        if descriptor.shape[1] == num_atoms:
            return np.transpose(descriptor, (1, 0, 2)).reshape(num_atoms, -1)
        if descriptor.shape[0] == num_atoms:
            return descriptor.reshape(num_atoms, -1)
    return descriptor.reshape(-1, descriptor.shape[-1])


def pooled_descriptor(descriptor: np.ndarray, pooling: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for pool in pooling:
        if pool == "mean":
            parts.append(descriptor.mean(axis=0))
        elif pool == "std":
            parts.append(descriptor.std(axis=0))
        elif pool == "max":
            parts.append(descriptor.max(axis=0))
        elif pool == "min":
            parts.append(descriptor.min(axis=0))
        else:  # pragma: no cover - argparse prevents this
            raise ValueError(f"unsupported pooling mode {pool!r}")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def get_mace_descriptor(calculator, atoms, args: argparse.Namespace) -> np.ndarray:
    kwargs: dict[str, Any] = {"invariants_only": args.invariants_only}
    if args.num_layers is not None:
        kwargs["num_layers"] = args.num_layers
    try:
        raw = calculator.get_descriptors(atoms, **kwargs)
    except TypeError:
        raw = calculator.get_descriptors(atoms)
    descriptor = descriptor_to_numpy(raw)
    descriptor = normalize_descriptor_shape(descriptor, len(atoms))
    return pooled_descriptor(descriptor, args.pooling)


def load_existing(path: Path) -> tuple[list[np.ndarray], list[float], list[str], list[str], list[int]]:
    if not path.exists():
        return [], [], [], [], []
    data = np.load(path, allow_pickle=True)
    return (
        [row.astype(np.float32, copy=False) for row in data["X"]],
        [float(value) for value in data["y"]],
        [str(value) for value in data["ids"]],
        [str(value) for value in data["formulas"]],
        [int(value) for value in data["n_atoms"]],
    )


def save_npz(
    path: Path,
    features: list[np.ndarray],
    targets: list[float],
    ids: list[str],
    formulas: list[str],
    n_atoms: list[int],
    metadata: dict[str, Any],
    compressed: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        raise ValueError("no descriptors to save")
    arrays = {
        "X": np.stack(features).astype(np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "ids": np.asarray(ids, dtype=str),
        "formulas": np.asarray(formulas, dtype=str),
        "n_atoms": np.asarray(n_atoms, dtype=np.int32),
        "metadata": np.asarray(json.dumps(metadata, indent=2)),
    }
    tmp = path.with_name(path.name + ".tmp.npz")
    if compressed:
        np.savez_compressed(tmp, **arrays)
    else:
        np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    output = Path(args.output) if args.output else default_output_path(args)
    if output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists. Use --resume to append or --overwrite to replace it."
        )

    entries = load_jarvis_entries(args)
    calculator = build_mace_calculator(args)

    features: list[np.ndarray]
    targets: list[float]
    ids: list[str]
    formulas: list[str]
    n_atoms: list[int]
    if args.resume:
        features, targets, ids, formulas, n_atoms = load_existing(output)
    else:
        features, targets, ids, formulas, n_atoms = [], [], [], [], []
    seen_ids = set(ids)

    metadata = {
        "dataset": args.dataset,
        "target": args.target,
        "mace_model": args.mace_model,
        "device": args.device,
        "default_dtype": args.default_dtype,
        "pooling": args.pooling,
        "invariants_only": args.invariants_only,
        "num_layers": args.num_layers,
        "max_samples": args.max_samples,
        "max_atoms": args.max_atoms,
        "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    start = time.perf_counter()
    failures = 0
    processed_this_run = 0
    total = len(entries)

    for position, entry in enumerate(entries, start=1):
        entry_id = str(entry.get("jid") or entry.get("id") or entry["_source_index"])
        if entry_id in seen_ids:
            continue
        try:
            atoms = jarvis_entry_to_ase(entry)
            descriptor = get_mace_descriptor(calculator, atoms, args)
            formula = reduced_formula(atoms.get_chemical_formula(mode="hill"))
        except Exception as exc:
            failures += 1
            message = f"[skip] {entry_id}: {exc}"
            if args.strict:
                raise RuntimeError(message) from exc
            print(message, flush=True)
            continue

        features.append(descriptor)
        targets.append(float(entry["_target_float"]))
        ids.append(entry_id)
        formulas.append(formula)
        n_atoms.append(len(atoms))
        seen_ids.add(entry_id)
        processed_this_run += 1

        if args.log_every > 0 and (
            processed_this_run % args.log_every == 0 or position == total
        ):
            elapsed = time.perf_counter() - start
            rate = processed_this_run / elapsed if elapsed > 0 else 0.0
            print(
                f"{processed_this_run} new / {len(features)} total descriptors "
                f"({position}/{total} rows, {rate:.3f} structures/s)",
                flush=True,
            )

        if args.save_every > 0 and processed_this_run % args.save_every == 0:
            save_npz(output, features, targets, ids, formulas, n_atoms, metadata, args.compressed)
            print(f"Checkpoint saved to {output}", flush=True)

    metadata["failures"] = failures
    metadata["num_descriptors"] = len(features)
    metadata["feature_dim"] = int(features[0].shape[0]) if features else None
    metadata["elapsed_seconds"] = time.perf_counter() - start
    save_npz(output, features, targets, ids, formulas, n_atoms, metadata, args.compressed)
    print(
        f"Done. Saved {len(features)} descriptors with dim={metadata['feature_dim']} to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
