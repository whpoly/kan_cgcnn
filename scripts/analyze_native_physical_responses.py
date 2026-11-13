"""Descriptive ALE of frozen four-way ensembles; no fitting or outer-label selection."""
import argparse
import gc
import itertools
import json
import os
import pickle
import traceback
from pathlib import Path
import run_native_fourway as engine  # registers the serialized B-spline layer
from run_modnet_official_matched import np, tf, MODData, file_hash, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    'mean_mass': 'ElementProperty|MagpieData mean AtomicWeight',
    'mean_bond_length': 'AverageBondLength|mean Average bond length',
    'mean_covalent_radius': 'ElementProperty|MagpieData mean CovalentRadius',
    'volume_per_atom': 'DensityFeatures|vpa',
    'mean_electronegativity': 'ElementProperty|MagpieData mean Electronegativity',
    'mean_en_difference': 'ElectronegativityDiff|mean EN difference',
    'coordination': 'CoordinationNumber|mean CN_VoronoiNN',
    'density': 'DensityFeatures|density',
}


def predict(model, raw):
    x = np.nan_to_num(raw)
    if model._scaler is not None:
        x = np.nan_to_num(model._scaler.transform(x), nan=-1)
    pieces = []
    for i in range(0, len(x), 256):
        pieces.append(np.asarray(model.model(x[i:i+256], training=False)).reshape(-1))
    return np.concatenate(pieces)


def ale_curve(effects, bins, edges):
    counts = np.bincount(bins, minlength=len(edges)-1)
    local = np.array([effects[bins == k].mean() if counts[k] else 0. for k in range(len(counts))])
    accumulated = np.cumsum(local)-.5*local
    accumulated -= np.average(accumulated, weights=counts)
    return dict(x=((edges[:-1]+edges[1:])/2).tolist(), ale=accumulated.tolist(),
                counts=counts.tolist(), edges=edges.tolist(),
                mean_abs_ale=float(np.average(abs(accumulated), weights=counts)))


def analyze_fold(run, out, arm, fold, banks, p, member_limit):
    directory = run/arm/f'fold-{fold}'
    done = out/arm/f'fold-{fold}.json'
    selection = engine.read(directory/'frozen-selection.json')
    complete = engine.read(directory/'complete.json')
    assert file_hash(directory/'frozen-selection.json') == complete['selection_sha256']
    assert file_hash(directory/'outer-predictions.npz') == complete['prediction_sha256']
    if done.exists():
        result = engine.read(done); assert result['protocol_fingerprint'] == digest(p)
        return result
    split = engine.read(engine.OLD/'matbench_phonons'/f'fold-{fold}'/'outer-split.json')
    frame = banks[arm.rsplit('_', 1)[0]]
    train = frame.loc[split['train_ids']]
    test = frame.loc[split['test_ids']]
    jobs = selection['members'][:member_limit] if member_limit else selection['members']
    with np.load(directory/'outer-predictions.npz') as z:
        reference = z['members'].copy()
    grids = {}; deltas = {}; selected = {}; member_strength = {}
    for label, column in FEATURES.items():
        values = train[column].values
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 3: continue
        bins = np.clip(np.searchsorted(edges, values, side='right')-1, 0, len(edges)-2)
        grids[label] = (edges, bins)
        deltas[label] = np.zeros(len(train)); selected[label] = 0; member_strength[label] = []
    max_error = 0.
    for n, job in enumerate(jobs):
        path = Path(job['directory'])/'model.pkl'
        assert file_hash(path) == selection['model_sha256'][job['directory']]
        with path.open('rb') as f: model = pickle.load(f)
        model._restore_model()
        cols = model.optimal_descriptors[:model.n_feat]
        raw = train[cols].values.copy()
        actual = predict(model, test[cols].values)
        error = float(np.max(abs(actual-reference[n])))
        assert error < .002, (arm, fold, n, error)
        max_error = max(max_error, error)
        for label, (edges, bins) in grids.items():
            column = FEATURES[label]
            effect = np.zeros(len(train))
            if column in cols:
                selected[label] += 1
                index = list(cols).index(column)
                low = raw.copy(); high = raw.copy()
                low[:, index] = edges[bins]; high[:, index] = edges[bins+1]
                preds = predict(model, np.concatenate([low, high]))
                effect = preds[len(train):]-preds[:len(train)]
            deltas[label] += effect/len(jobs)
            member_strength[label].append(ale_curve(effect, bins, edges)['mean_abs_ale'])
        write_json(out/'status.json', dict(state='analyzing', pid=os.getpid(), arm=arm,
                   fold=fold, member=n+1, members=len(jobs)))
        del model; tf.keras.backend.clear_session(); gc.collect()
    curves = {}
    for label, (edges, bins) in grids.items():
        curve = ale_curve(deltas[label], bins, edges)
        curve.update(column=FEATURES[label], selected_members=selected[label],
                     member_strength=member_strength[label],
                     central_domain=np.quantile(train[FEATURES[label]], [.05, .95]).tolist())
        curves[label] = curve
    result = dict(arm=arm, fold=fold, members=len(jobs), curves=curves,
                  max_prediction_replay_error=max_error, protocol_fingerprint=digest(p))
    write_json(done, result)
    return result


def summarize(out, rows, p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = []
    for arm in engine.ARMS:
        group = [r for r in rows if r['arm'] == arm]
        if len(group) != 5: continue
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for ax, label in zip(axes.flat, FEATURES):
            curves = [r['curves'][label] for r in group if label in r['curves']]
            if len(curves) != 5: continue
            lower = max(c['central_domain'][0] for c in curves)
            upper = min(c['central_domain'][1] for c in curves)
            correlations = []
            if upper > lower:
                grid = np.linspace(lower, upper, 100)
                values = [np.interp(grid, c['x'], c['ale']) for c in curves]
                correlations = [float(np.corrcoef(a,b)[0,1]) for a,b in itertools.combinations(values,2)
                                if np.std(a)>1e-8 and np.std(b)>1e-8]
            for i,c in enumerate(curves): ax.plot(c['x'], c['ale'], label=f'Fold {i+1}')
            ax.axhline(0, color='gray', linewidth=.5); ax.set_title(label)
            ax.set_xlabel('Descriptor value (original scale)'); ax.set_ylabel('ALE (cm^-1)')
            summary.append(dict(arm=arm, feature=label,
                mean_abs_ale_by_fold=[c['mean_abs_ale'] for c in curves],
                mean_abs_ale=float(np.mean([c['mean_abs_ale'] for c in curves])),
                pairwise_curve_correlations=correlations,
                median_curve_correlation=float(np.median(correlations)) if correlations else None,
                common_central_domain=[lower,upper],
                selected_members_by_fold=[c['selected_members'] for c in curves]))
        axes.flat[0].legend(); fig.suptitle(arm+' | frozen ensemble ALE | descriptive, not causal')
        fig.tight_layout(); fig.savefig(out/(arm+'-physical-ale.png'), dpi=160); plt.close(fig)
    write_json(out/'summary.json', dict(complete=len(rows)==20, smoke=p['member_limit']>0,
        rows=summary, protocol_fingerprint=digest(p),
        interpretation='ALE amplitude is model response, not predictive importance or causal effect. '
        'Correlated descriptors can substitute for each other. No formula was selected using outer labels.'))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--member-limit',type=int,default=0)
    ap.add_argument('--fold-limit',type=int,default=5); args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(exist_ok=True,parents=True); run=args.run.resolve()
    p=dict(run=str(run),source_sha256=file_hash(__file__),features=FEATURES,
        method='10-quantile-bin first-order ALE, frozen ensemble, common outer-training X background',
        member_limit=args.member_limit,fold_limit=args.fold_limit,
        benchmark_audit_sha256=file_hash(run/'artifact-audit.json'),
        limitations=['Physical descriptor interventions may not correspond to realizable structures.',
          'ALE is descriptive of the full model, not a discovered physical law.',
          'Cross-fold curves use overlapping training samples; not independent significance tests.',
          'Only fixed candidate physical descriptors are examined; embedding effects remain unexplained.'])
    if (out/'protocol.json').exists(): assert engine.read(out/'protocol.json')==p
    write_json(out/'protocol.json',p)
    # Loading banks checks original dataset hashes; its audit is written only to this new directory.
    banks,_=engine.load_banks(out)
    rows=[]
    for arm in engine.ARMS:
        for fold in range(args.fold_limit): rows.append(analyze_fold(run,out,arm,fold,banks,p,args.member_limit))
    summarize(out,rows,p)
    write_json(out/'status.json',dict(state='complete',complete=len(rows)==20,pid=os.getpid(),smoke=args.member_limit>0))


if __name__=='__main__':
    try: main()
    except BaseException:
        if '--output' in __import__('sys').argv:
            out=Path(__import__('sys').argv[__import__('sys').argv.index('--output')+1])
            write_json(out/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
