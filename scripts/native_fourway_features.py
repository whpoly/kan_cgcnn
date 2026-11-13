"""No-SISSO additive banks and resumable extension of native cross-NMI."""
from run_modnet_official_matched import np, pd, MODData, SafePool, file_hash, digest, write_json, set_seed
from modnet.preprocessing import compute_mi, nmi_target, get_rr_p_parameter_default, get_rr_c_parameter_default
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'benchmarks/official-aligned-20261021'
CSV = ROOT/'benchmarks/mattervial-phonons-20261023/source/extracted/featurized_datasets/matbench_phonons_featurizedMM2020Struct_mattervial.csv'
CSV_SHA = '6369c2322436adfaa29834b3e456192c73eab3d7da159f14c0a6102d91d0b535'
ORIGINAL = OLD/'precomputed/matbench_phonons_moddata.pkl.gz'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_banks(out):
    if file_hash(CSV) != CSV_SHA: raise ValueError('Changed MatterVial CSV')
    data = MODData.load(ORIGINAL)
    assert list(data.df_targets.index) == [f'id{i}' for i in range(1265)]
    mm = data.df_featurized.replace([np.inf, -np.inf], np.nan).fillna(0)
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    prefixes = ('MEGNet_OFMEncoded_v1_', 'MVL16_', 'MVL32_', 'ROOST_mpgap_', 'ORB_v3_')
    extras = sorted(c for c in header if c.startswith(prefixes) and 'SISSO' not in c.upper())
    ids = [f'mb-phonons-{i+1:04d}' for i in range(1265)]
    mv = pd.read_csv(CSV, usecols=['material_id', 'target']+extras).set_index('material_id').loc[ids]
    np.testing.assert_allclose(mv.target.values, data.df_targets.values[:, 0], rtol=0, atol=1e-8)
    mv = mv.drop(columns='target'); mv.index = mm.index
    mv = mv.replace([np.inf, -np.inf], np.nan).fillna(0)
    enhanced = pd.concat([mm, mv], axis=1)
    assert enhanced.columns.is_unique and not any('SISSO' in c.upper() for c in enhanced)
    pd.testing.assert_frame_equal(enhanced[mm.columns], mm)
    write_json(out/'data-audit.json', dict(mm_features=len(mm.columns), added_features=len(extras),
        enhanced_features=len(enhanced.columns), excluded_sisso=sum('SISSO' in c.upper() for c in header),
        extras=extras, identical_mm_values=True, targets_match=True, ids=ids))
    return dict(mm=mm, mm_mv=enhanced), data.df_targets


def make_data(frame, target):
    return MODData(materials=[None]*len(frame), targets=target.values,
        target_names=list(target.columns), structure_ids=list(frame.index), df_featurized=frame)


def fast_native_ranking(cross, target):
    """Native RR formula; minimum over selected scores equals max redundancy.

    For nonnegative NMI and positive p, min_j t/(r_j**p+c) = t/(max_j r_j**p+c).
    This eliminates huge repeated pandas copies while preserving tie ordering.
    """
    columns = [c for c in cross.index if c in target.index]
    values = cross.loc[columns, columns].values.astype(float)
    target = target.loc[[c for c in target.index if c in columns]].replace([np.inf, -np.inf, np.nan], 0)
    assert np.isfinite(values).all() and (values >= 0).all() and (target >= 0).all()
    first = target.nlargest(1).index[0]
    scores = target.reindex(columns).values.astype(float)
    selected = [columns.index(first)]; available = np.ones(len(columns), dtype=bool)
    redundancy = values[:, selected[0]].copy(); available[selected[0]] = False
    for n in range(len(columns)-1):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = scores/(redundancy**get_rr_p_parameter_default(n)+get_rr_c_parameter_default(n))
        # c(0)=0: native pandas idxmax skips a 0/0 candidate, while retaining
        # positive/0 as +inf. numpy argmax alone would incorrectly pick NaN.
        if np.isnan(ratio[available]).all(): raise ValueError('Native RR has no defined candidate score')
        ratio[np.isnan(ratio)] = -np.inf
        ratio[~available] = -np.inf
        index = int(np.argmax(ratio)); selected.append(index); available[index] = False
        redundancy = np.maximum(redundancy, values[:, index])
    return [columns[i] for i in selected]


def entropy_job(job):
    x = np.load(job['matrix'], mmap_mode='r')
    return [(i, float(compute_mi(x[:, i], x[:, i], random_state=job['seed'])[0])) for i in job['indices']]


def cross_row(job):
    path = Path(job['path']); sig = digest(job)
    if path.exists():
        with np.load(path) as z:
            assert str(z['signature']) == sig
            return str(path)
    x = np.load(job['matrix'], mmap_mode='r'); i = job['i']
    js = np.array(job['js'], dtype=int)
    values = np.array([compute_mi(x[:, i], x[:, j], random_state=job['seed'])[0] /
        (.5*(job['entropy_i']+h)) for j, h in zip(js, job['entropy_js'])])
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, signature=sig, js=js, values=values)
    temp.replace(path)
    return str(path)


def extend_ranking(data, old_data, directory, seed, n_jobs, progress):
    """Preserve the original MM-MM block; compute missing pairs on outer train.

    Native precomputed-only selection would otherwise silently exclude MV.
    Entropy filtering and every missing-pair MI use original MODNet functions.
    """
    directory.mkdir(parents=True, exist_ok=True)
    frame = data.df_featurized
    matrix_path = directory/'training-features.npy'
    if not matrix_path.exists(): np.save(matrix_path, frame.values)
    np.testing.assert_array_equal(np.load(matrix_path, mmap_mode='r'), frame.values)
    matrix_sha = file_hash(matrix_path)
    columns = list(frame.columns)
    base = old_data.cross_nmi
    known = set(base.columns) & set(columns)
    entropy_file = directory/'entropy.json'
    signature = digest(dict(matrix=matrix_sha, columns=columns, seed=seed))
    if entropy_file.exists():
        stored = read(entropy_file); assert stored['signature'] == signature
        entropy = np.array(stored['entropy'])
    else:
        jobs = [dict(matrix=str(matrix_path), seed=seed, indices=list(range(i, min(i+32, len(columns)))))
                for i in range(0, len(columns), 32)]
        entropy = np.zeros(len(columns)); done = 0
        with SafePool(n_jobs) as pool:
            for result in pool.imap_unordered(entropy_job, jobs):
                for i, h in result: entropy[i] = h
                done += len(result); progress('feature_entropy', completed=done, total=len(columns))
        write_json(entropy_file, dict(signature=signature, entropy=entropy))
    # The MM admissible set comes from the original external cross-NMI. Apply
    # native entropy filtering only to newly introduced feature columns.
    mm_columns = set(old_data.df_featurized.columns)
    keep = [i for i, c in enumerate(columns) if c in known or
            (c not in mm_columns and entropy[i] >= .2 and np.ptp(frame.iloc[:, i].values) >= 1e-16)]
    kept = [columns[i] for i in keep]
    assert any(c not in mm_columns for c in kept), 'Enhanced features were lost'
    rows_dir = directory/'rows'; rows_dir.mkdir(exist_ok=True)
    rows = []
    for pos, i in enumerate(keep):
        js = [j for j in keep[pos+1:] if not (columns[i] in known and columns[j] in known)]
        if js:
            rows.append(dict(matrix=str(matrix_path), matrix_sha=matrix_sha, i=i, js=js,
                seed=seed, entropy_i=float(entropy[i]), entropy_js=entropy[js].tolist(),
                path=str(rows_dir/f'row-{i:04d}.npz')))
    completed = 0
    with SafePool(n_jobs) as pool:
        for _ in pool.imap_unordered(cross_row, rows):
            completed += 1; progress('feature_cross_nmi', completed=completed, total=len(rows))
    cross = pd.DataFrame(np.eye(len(kept)), index=kept, columns=kept)
    shared = [c for c in kept if c in known]
    cross.loc[shared, shared] = base.loc[shared, shared].values
    for row in rows:
        with np.load(row['path']) as z:
            a = columns[row['i']]; bs = [columns[i] for i in z['js']]
            cross.loc[a, bs] = z['values']; cross.loc[bs, a] = z['values']
    set_seed(seed)
    target = nmi_target(frame, data.df_targets, 'regression', random_state=seed).iloc[:, 0]
    data.cross_nmi = cross; data.target_nmi = target
    data.optimal_features = fast_native_ranking(cross, target)
    data.optimal_features_by_target = {data.target_names[0]: data.optimal_features}
    write_json(directory/'selection-audit.json', dict(seed=seed, training_ids=list(frame.index),
        columns=columns, selected_features=data.optimal_features, native_mm_block_preserved=True,
        new_feature_count=sum(c not in mm_columns for c in data.optimal_features),
        matrix_sha=matrix_sha, scope='outer train; shared by MODNet and KAN; no RFE, no SISSO'))
    return data
