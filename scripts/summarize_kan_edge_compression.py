"""Summarize audited fixed-policy compression; no model or policy selection."""
import argparse
import hashlib
import json
from pathlib import Path
from modnet_runtime import configure_official_runtime
configure_official_runtime()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    audit = read(out / 'artifact-audit.json')
    assert audit['passed'] and audit['complete'] and audit['model_recipes'] == 500
    assert hashlib.sha256((out / 'analysis.json').read_bytes()).hexdigest() == audit['analysis_sha256']
    analysis = read(out / 'analysis.json')
    summary = {'analysis_sha256': audit['analysis_sha256'], 'arms': {},
               'scope': 'Fixed degree 1/2/3 whole-network polynomial replacement; training-teacher CV selects the degree or original-spline fallback.',
               'limits': analysis['limitations'] + ['Coefficient counts are not disk storage measurements.',
                   'An unchanged MAE from original-spline fallback is not successful compression.']}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, arm, label in zip(axes, ['mm_kan', 'mm_mv_kan'], ['MM + KAN', 'MM + MatterVial + KAN']):
        rows = [r for r in analysis['rows'] if r['arm'] == arm]
        assert len(rows) == 5
        recipes = [read(p) for p in sorted((out / arm).glob('fold-*/member-*/frozen-recipe.json'))]
        assert len(recipes) == 250
        ratios = np.array([[s['mean_teacher_mae'] / r['tolerance'] for s in r['cv']] for r in recipes])
        counts = {str(degree): sum(r['degree'] == degree for r in recipes) for degree in [None, 1, 2, 3]}
        original = sum(r['original_entries'] for r in rows)
        compressed = sum(r['compressed_entries'] for r in rows)
        summary['arms'][arm] = dict(fold_mae=[r['mae'] for r in rows],
            original_fold_mae=[r['teacher_mae'] for r in rows], means=analysis['means'][arm],
            degree_counts=counts, original_entries=original, compressed_entries=compressed,
            coefficient_reduction_fraction=1-compressed/original,
            training_teacher_cv_error_over_tolerance={str(d): dict(min=float(ratios[:, d-1].min()),
                median=float(np.median(ratios[:, d-1])), max=float(ratios[:, d-1].max())) for d in [1, 2, 3]})
        ax.boxplot([ratios[:, i] for i in range(3)], labels=['1', '2', '3'], showfliers=False)
        ax.axhline(1, color='#b43b3b', linestyle='--', label='Acceptance threshold')
        ax.set_yscale('log')
        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel('Training-teacher CV MAE / tolerance')
        ax.set_title(f'{label}\nOriginal-spline fallback: {counts["None"]}/250')
        ax.legend(fontsize=8)
    fig.suptitle('Fixed-policy edge compression: approximation check, not target prediction')
    fig.savefig(out / 'compression-validation.png', dpi=170)
    plt.close(fig)
    (out / 'interpretation-verdict.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary['arms']))


if __name__ == '__main__':
    main()
