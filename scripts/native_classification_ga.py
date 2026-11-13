"""Versioned native GA adapter: stratified CV, positive monotone AUC fitness."""
from run_native_fourway import *
# Worker function is imported lazily to avoid a circular module initialization.
def run_fit(job):
    from run_native_matbench_classification_v1 import run_fit as implementation
    return implementation(job)

class ClassificationGA(FitGenetic):
    def __init__(self, data, directory, cache, p, arm, fold, seed, progress, n_jobs):
        super().__init__(data)
        self.directory=directory; self.cache=cache; self.training_sha=file_hash(cache)
        self.p=p; self.arm=arm; self.fold=fold; self.seed=seed; self.progress=progress
        self.generation=0; self.job_by_path={}; self.records=[]; self.n_jobs=n_jobs

    def _init_run(self, n_jobs=None): self.pool=SafePool(n_jobs or self.n_jobs)

    def function_fitness(self, pop, n_jobs, nested=5, val_fraction=.1, multi_label=False, fast=False):
        assert not multi_label and nested
        splits=list(matbench_kfold_splits(self.train_data, n_splits=nested, classification=True))
        jobs=[]; fp=digest(self.p)
        for candidate, individual in enumerate(pop):
            for inner,(tr,va) in enumerate(splits):
                ids=np.array(sample_ids(self.train_data))
                directory=self.directory/'fits'/f'generation-{self.generation:02d}-candidate-{candidate:02d}-inner-{inner}'
                job=dict(directory=str(directory), genes=individual.genes.copy(), seed=self.seed+self.generation*10000+candidate*100+inner,
                    arm=self.arm, variant='modnet' if self.arm.endswith('modnet') else 'bspline',
                    fold=self.fold, generation=self.generation, candidate=candidate, inner=inner,
                    train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(), smoke=self.p['smoke'],
                    training_cache=str(self.cache), training_sha256=self.training_sha,
                    protocol_path=self.p['protocol_path'], protocol_fingerprint=fp,
                    run_fingerprint=digest(dict(protocol=fp,training_data=self.training_sha)))
                jobs.append(job); self.job_by_path[str(directory/'model.pkl')]=job
        losses=np.empty((len(pop),nested)); positions={j['directory']:i for i,j in enumerate(jobs)}
        for done,result in enumerate(self.pool.imap_unordered(run_fit,jobs),1):
            candidate,inner=divmod(positions[result['directory']],nested)
            losses[candidate,inner]=result['selection_loss']; self.records.append(result)
            self.progress('training', arm=self.arm, fold=self.fold, generation=self.generation,
                generation_completed=done, generation_total=len(jobs), completed=len(self.records),
                total=self.p['fits_per_arm_fold'], last_selection_loss=result['selection_loss'])
            print('FIT',self.arm,self.fold,self.generation,done,'/',len(jobs),result['selection_loss'],flush=True)
        models=[]; individuals=[]
        for candidate,source in enumerate(pop):
            refs=[ModelReference(str(Path(jobs[candidate*nested+i]['directory'])/'model.pkl'),
                source.targets,source.weights,source.num_classes) for i in range(nested)]
            models.append(EnsembleMODNetModel(modnet_models=refs))
            ind=copy.copy(source); ind.model=refs[-1]; ind.val_loss=losses[candidate,-1]; individuals.append(ind)
        write_json(self.directory/f'generation-{self.generation:02d}.json',dict(
            genes=[i.genes for i in pop],selection_loss=losses,mean_selection_loss=losses.mean(axis=1)))
        self.generation+=1
        return losses.mean(axis=1),np.array(models),np.array(individuals)

