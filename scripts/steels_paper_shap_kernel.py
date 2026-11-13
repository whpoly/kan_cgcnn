"""Paper-informed SHAP settings; explainer choice is an explicit reconstruction assumption."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.deps/steels-shap-v3'))
import numpy as np
import shap


def importance(x, predict, seed, smoke=False):
    """Use only supplied inner-training rows for explanation and background."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(x), min(len(x), 8 if smoke else 300), replace=False)
    background = x[indices]
    np.random.seed(seed)
    explainer = shap.KernelExplainer(predict, background, link='identity')
    values = np.asarray(explainer.shap_values(
        background, nsamples=32 if smoke else 500, l1_reg=0, silent=True))
    assert values.shape == background.shape and np.isfinite(values).all()
    residual = np.max(np.abs(values.sum(axis=1) + float(explainer.expected_value)
                             - predict(background)))
    assert residual < 1e-3, residual
    return np.abs(values).mean(axis=0), indices, dict(
        library='shap', version=shap.__version__, explainer='KernelExplainer',
        explainer_is_reconstruction_assumption=True,
        samples=len(indices), sample_cap=8 if smoke else 300,
        nsamples=32 if smoke else 500, l1_reg=0,
        background='same inner-training rows; no validation or outer-test rows',
        max_additivity_residual=float(residual))
