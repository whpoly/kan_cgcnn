import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path("scripts").resolve()))
import run_steel396_shap_pair as run
import msvcrt
out = run.BASE / "formal"
with (out/"run.lock").open("r+b") as lock:
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
with run.np.load(run.INPUT / "inputs.npz", allow_pickle=False) as z:
    pos = z["train_positions_2"]
    df = run.pd.DataFrame(z["X"][pos], columns=z["columns"], index=z["sample_ids"][pos])
    train = run.make_data(df,z["y"][pos])
inner = run.fresh_pairs(train,5)[1][0]
sel = run.read(out/"fold-2/native/selection.json")
def measure(x, predict, seed, smoke):
    np=run.np
    z=np.asarray(x,dtype=np.float64)
    rows=np.random.RandomState(seed).choice(len(z),min(len(z),300),replace=False)
    b=z[rows]
    batched=predict(b)
    single=np.concatenate([predict(row[None,:]) for row in b])
    print("probe: batch disagreement",float(np.abs(single-batched).max()), "MPa",flush=True)
    raise StopIteration
run.importance=measure
with TemporaryDirectory() as tmp:
    try:
        run.get_ranking(Path(tmp),1,inner,sel["members"],sel["fp"],260902+200000+1+524,False)
    except StopIteration:
        pass
