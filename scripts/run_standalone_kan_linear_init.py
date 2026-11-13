"""20-fit initialization-only ablation, declared after the first screen seed.

Uses the frozen trainer in a separate process with an explicit builder override.
No files from the concurrently running screen are modified.
"""
import run_standalone_kan_screen as engine
from standalone_kan_models import build_model as original_builder,SplineEdgeLayer,COUNTS,np,tf
from run_modnet_official_matched import SafePool,write_json,digest,file_hash
from pathlib import Path
import argparse
import os
import time
import traceback


def linear_initialized_model(bank,variant,seed):
    if variant!='kan':raise ValueError('Only the unregularized KAN is in this ablation')
    model=original_builder(bank,variant,seed)
    knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3]
    greville=np.array([knots[i+1:i+4].mean() for i in range(8)])
    for index,layer in enumerate(l for l in model.layers if isinstance(l,SplineEdgeLayer)):
        nin,nout,_=layer.coefficients.shape
        rng=np.random.RandomState(seed+1900+index)
        limit=np.sqrt(6/(int(nin)+int(nout)))
        weights=rng.uniform(-limit,limit,(int(nin),int(nout)))
        # Hidden input is 0.5*tanh(h). A factor of two makes the initial
        # edge compute w*tanh(h), avoiding an unintended half-scale per layer.
        gain=2. if layer.hidden_coordinates else 1.
        coefficients=gain*weights[...,None]*greville
        coefficients+=rng.normal(0,.001/np.sqrt(int(nin)),size=coefficients.shape)
        layer.coefficients.assign(coefficients.astype('float32'))
        layer.bias.assign(tf.zeros_like(layer.bias))
    return model


# Process-local dependency injection is part of this experiment's saved source.
engine.build_model=linear_initialized_model


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=2);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    source=engine.ROOT/'benchmarks/standalone-kan-screen-20261027/protocol.json'
    p=engine.read(source)
    for name,value in p['source_sha256'].items():
        if file_hash(engine.ROOT/'scripts'/name)!=value:raise ValueError('Changed frozen source')
    protocol=dict(kind='standalone KAN initialization-only follow-up',parent_protocol_sha256=file_hash(source),
        parent_protocol=p,source_sha256={**p['source_sha256'],Path(__file__).name:file_hash(__file__)},
        adaptation='Added after observing first-seed screening and activation diagnostics; not a preplanned independent confirmation',
        change='Initialize each edge near a linear function in its spline coordinate, with small coefficient noise; no architecture changes',
        initializer='Glorot uniform edge slopes; Greville linear reproduction; hidden-edge slope gain 2; noise SD .001/sqrt(fan_in)',
        declared_at=engine.now(),banks=['enhanced','physical'],variant='kan',seeds=p['seeds'],
        learning_rate=p['learning_rate'],max_epochs=p['max_epochs'],fits=20,outer_test_evaluated=False,
        search_budget_limitation='Adds one KAN candidate, no added MLP search; disclose all configurations and do not claim equal family-level tuning')
    path=out/'protocol.json'
    if path.exists():
        saved=engine.read(path);protocol['declared_at']=saved['declared_at']
        if digest(saved)!=digest(protocol):raise ValueError('Follow-up protocol changed')
    write_json(path,protocol);fp=digest(protocol)
    def status(state,**kwargs):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=engine.now(),**kwargs))
    try:
        status('training',completed=0,total=20)
        jobs=[dict(directory=str(out/'fits'/bank/'kan'/f'seed-{s}-inner-{i}'),bank=bank,variant='kan',repeat=s,inner=i,
            seed=seed+i,lr=p['learning_rate'],max_epochs=p['max_epochs'],expected_parameters=COUNTS[bank]['kan'],
            input=p['source_inputs'][i],run_fingerprint=fp)
            for s,seed in enumerate(p['seeds']) for i in range(5) for bank in ('enhanced','physical')]
        records=[]
        with SafePool(args.n_jobs) as pool:
            for r in pool.imap_unordered(engine.train_job,jobs):
                records.append(r);status('training',completed=len(records),total=20,last_bank=r['bank'],last_mae=r['validation_mae'])
                write_json(out/'progress-summary.json',dict(completed=len(records),rows=[dict(bank=bank,repeat=s,
                    completed=sum(r['bank']==bank and r['repeat']==s for r in records),
                    mean_mae=float(np.mean([r['validation_mae'] for r in records if r['bank']==bank and r['repeat']==s])))
                    for s in range(2) for bank in ('enhanced','physical') if any(r['bank']==bank and r['repeat']==s for r in records)]))
                print('FIT',len(records),'/20',r['bank'],'seed',r['repeat'],'inner',r['inner'],'MAE',r['validation_mae'],
                    'epoch',r['best_epoch'],'/',r['epochs'],flush=True)
        status('complete',completed=20,total=20,wall_seconds=time.perf_counter()-start,outer_test_evaluated=False)
        print('COMPLETE',out,flush=True)
    except Exception:status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
