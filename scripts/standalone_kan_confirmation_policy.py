"""Predeclared split, tuning and go/no-go rules; no model or data-file imports."""
import numpy as np

BANKS = ('enhanced', 'physical')
VARIANTS = ('kan', 'matched_mlp')
LRS = (.0005, .001, .002)
TUNE_SEEDS = (480281, 481281)
CONFIRM_SEEDS = (580281, 581281)
OUTER_SEEDS = (680281, 681281)
SPLIT_SEED = 380281
SUBSPLIT_SEED = 380283
GATE = dict(primary_bank='enhanced', min_relative_improvement=.02,
            min_total_wins=7, min_wins_per_seed=3, min_fold_mean_wins=4,
            require_both_seed_means_better=True)


def make_splits(ids):
    ids = sorted(ids)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs')
    folds = np.array_split(np.random.RandomState(SPLIT_SEED).permutation(len(ids)), 5)
    result = []
    for i, indices in enumerate(folds):
        holdout = {ids[j] for j in indices}
        train = [x for x in ids if x not in holdout]
        perm = np.random.RandomState(SUBSPLIT_SEED+i).permutation(len(train))
        valid = {train[j] for j in perm[:int(np.ceil(.2*len(train))) ]}
        result.append(dict(inner=i, train_ids=train, validation_ids=sorted(holdout),
            tuning_train_ids=[x for x in train if x not in valid], tuning_validation_ids=sorted(valid)))
    return result


def choose(records, expected_per_lr):
    """Only tuning records accepted; deterministically select LR then fixed epochs."""
    if len(records) != len(LRS)*expected_per_lr:
        raise ValueError('Incomplete tuning budget')
    rows=[]
    for lr in LRS:
        group=sorted([r for r in records if r['lr']==lr],key=lambda r:(r.get('inner',0),r.get('repeat',0)))
        if len(group)!=expected_per_lr:
            raise ValueError('Unequal LR budget')
        values=np.array([r['validation_mae'] for r in group])
        if not np.isfinite(values).all():raise ValueError('Nonfinite tuning score')
        rows.append(dict(lr=lr, mean_mae=float(values.mean()),
            epochs=int(np.floor(np.median([r['best_epoch'] for r in group])+.5)),
            best_epochs=[r['best_epoch'] for r in group]))
    winner=min(rows,key=lambda r:(r['mean_mae'],r['lr']))
    return dict(lr=winner['lr'],epochs=winner['epochs'],candidates=rows)


def evaluate_gate(records):
    expected={(b,v,s,i) for b in BANKS for v in VARIANTS for s in range(2) for i in range(5)}
    keys=[(r['bank'],r['variant'],r['repeat'],r['inner']) for r in records]
    if len(keys)!=40 or set(keys)!=expected:raise ValueError('Incomplete confirmation')
    index={key:r for key,r in zip(keys,records)}
    rows={}
    for bank in BANKS:
        kan=np.array([[index[(bank,'kan',s,i)]['validation_mae'] for i in range(5)] for s in range(2)])
        mlp=np.array([[index[(bank,'matched_mlp',s,i)]['validation_mae'] for i in range(5)] for s in range(2)])
        if not np.isfinite(kan).all() or not np.isfinite(mlp).all():raise ValueError('Nonfinite confirmation')
        delta=kan-mlp
        relative=1-kan.mean()/mlp.mean()
        criteria=dict(relative_improvement=bool(relative>=GATE['min_relative_improvement']),
            total_wins=bool((delta<0).sum()>=GATE['min_total_wins']),
            each_seed_wins=bool(np.all((delta<0).sum(axis=1)>=GATE['min_wins_per_seed'])),
            fold_mean_wins=bool((delta.mean(axis=0)<0).sum()>=GATE['min_fold_mean_wins']),
            both_seed_means=bool(np.all(delta.mean(axis=1)<0)))
        rows[bank]=dict(kan_mae=kan.tolist(),mlp_mae=mlp.tolist(),delta=delta.tolist(),
            kan_seed_mae=kan.mean(axis=1).tolist(),mlp_seed_mae=mlp.mean(axis=1).tolist(),
            relative_improvement=float(relative),total_wins=int((delta<0).sum()),
            seed_wins=(delta<0).sum(axis=1).tolist(),fold_mean_wins=int((delta.mean(axis=0)<0).sum()),
            criteria=criteria,passed=all(criteria.values()))
    return dict(rule=GATE,passed=rows[GATE['primary_bank']]['passed'],rows=rows,
        meaning='Predeclared practical stability gate, not a statistical significance test; secondary bank cannot rescue primary')
