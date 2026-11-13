"""ID-only official-fold validation; frozen aggregate metric definitions."""
import numpy as np


def official_ids(manifest):
    train=[f'mb-phonons-{int(i)+1:04d}' for i in manifest['train_positions']]
    test=[f'mb-phonons-{int(i)+1:04d}' for i in manifest['test_positions']]
    if manifest['train_ids']!=['id'+str(i) for i in manifest['train_positions']]:raise ValueError('Training position/ID mismatch')
    if manifest['test_ids']!=['id'+str(i) for i in manifest['test_positions']]:raise ValueError('Test position/ID mismatch')
    if test!=manifest['official_test_ids']:raise ValueError('Official test ID mismatch')
    if len(train)!=1012 or len(test)!=253 or len(set(train))!=1012 or len(set(test))!=253 or set(train)&set(test):
        raise ValueError('Invalid official fold')
    if set(train+test)!={f'mb-phonons-{i:04d}' for i in range(1,1266)}:raise ValueError('Wrong universe')
    return sorted(train),test


def validate_fivefold(folds):
    if len(folds)!=5:raise ValueError('Five outer folds required')
    ids=[x for f in folds for x in f['test_ids']]
    if len(ids)!=len(set(ids)) or set(ids)!={f'mb-phonons-{i:04d}' for i in range(1,1266)}:
        raise ValueError('Outer test folds do not partition the dataset')


def metrics(kan,mlp):
    kan,mlp=np.asarray(kan,float),np.asarray(mlp,float)
    if kan.shape!=(5,) or mlp.shape!=(5,) or not np.isfinite(kan).all() or not np.isfinite(mlp).all():
        raise ValueError('Five finite fold scores required')
    return dict(kan_mean=float(kan.mean()),mlp_mean=float(mlp.mean()),kan_fold_sample_sd=float(kan.std(ddof=1)),
        mlp_fold_sample_sd=float(mlp.std(ddof=1)),absolute_improvement=float(mlp.mean()-kan.mean()),
        relative_improvement=float(1-kan.mean()/mlp.mean()),wins=int((kan<mlp).sum()),
        fold_absolute_improvement=(mlp-kan).tolist(),fold_relative_improvement=(1-kan/mlp).tolist())
