"""Reconstruct upstream FitGenetic's fixed sample cap without changing its RNG."""
from run_modnet_official_matched import np
from sklearn.model_selection import KFold, StratifiedKFold
from modnet.matbench.benchmark import MATBENCH_SEED

def partition_ids(index, targets, seed, nested=5, classification=False):
    ids=np.asarray(index)
    positions=np.random.RandomState(seed).permutation(len(ids))[:5000]
    sampled=ids[positions]
    splitter=(StratifiedKFold if classification else KFold)(n_splits=nested,shuffle=True,random_state=MATBENCH_SEED)
    labels=np.asarray(targets).reshape(-1)[positions]
    folds=[(sampled[tr].tolist(),sampled[va].tolist()) for tr,va in splitter.split(positions,y=labels)]
    return sampled.tolist(),folds
