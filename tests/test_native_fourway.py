import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_native_fourway as run
from native_fourway_models import NativeSplineModel, matched_widths, dense_count, spline_count
from native_fourway_features import fast_native_ranking, cross_row
from modnet.preprocessing import get_features_dyn, compute_mi

np,pd,tf=run.np,run.pd,run.tf


class NativeFourwayTest(unittest.TestCase):
    def test_native_initial_range_and_random_stream_unchanged(self):
        for seed in range(20):
            run.set_seed(seed)
            original=run.NativeIndividual(963,{'target':0},False)
            tail=np.random.rand()
            run.set_seed(seed)
            adapted=run.IntegerBoundIndividual(963,{'target':0},False)
            self.assertEqual(original.genes,adapted.genes)
            self.assertEqual(tail,np.random.rand())
        for seed in range(20):
            run.set_seed(seed)
            individual=run.IntegerBoundIndividual(3359,{'target':0},False)
            n=individual.genes['n_feat']
            self.assertTrue(1<=n<=3359)
            self.assertEqual(int(np.sqrt(n))**2,n)
            child=individual.crossover(individual)
            self.assertEqual(child.genes,individual.genes)

    def test_training_rules_identical_to_native(self):
        individual=run.IntegerBoundIndividual(100,{'target':0},False,targets=[[['target']]],weights={'target':1})
        with patch.object(run.MODNetModel,'fit') as fit, patch.object(run.MODNetModel,'evaluate',return_value=1):
            individual.evaluate('train','val');old=fit.call_args.kwargs.copy()
            run.fit_spline(individual,'train','val','bspline',seed=22);new=fit.call_args.kwargs.copy()
        for settings in (old,new):
            c=settings.pop('callbacks')[0]
            self.assertEqual(c.monitor,'loss');self.assertEqual(c.patience,30)
            self.assertEqual(abs(c.min_delta),.001);self.assertFalse(c.restore_best_weights)
        self.assertEqual(old,new)
        self.assertIs(NativeSplineModel.fit,run.MODNetModel.fit)
        self.assertIs(NativeSplineModel.evaluate,run.MODNetModel.evaluate)

    def test_parameter_mapping_and_serialization(self):
        for n,widths in ((256,(96,72,36,36)),(10,(32,8,2,0)),(190,(96,72,36,36)),(3359,(320,320,320,320))):
            a,b=matched_widths(n,widths)
            self.assertLessEqual(abs(spline_count(n,a,b)/dense_count(n,widths)-1),.05)
        for scale in ('minmax','standard'):
            tf.keras.backend.clear_session()
            model=NativeSplineModel(targets=[[['target']]],weights={'target':1},n_feat=10,
                num_neurons=[[32],[8],[2],[0]],spline_seed=7,input_scale=scale)
            self.assertFalse(any(isinstance(l,tf.keras.layers.Dense) for l in model.model.layers))
            x=tf.constant(np.random.RandomState(4).normal(size=(4,10)),dtype=tf.float32)
            before=model.model(x).numpy();model._make_picklable();model._restore_model()
            np.testing.assert_array_equal(before,model.model(x).numpy())

    def test_rr_formula_preserves_native_ranking_and_ties(self):
        for seed in range(5):
            rng=np.random.RandomState(seed);a=rng.uniform(size=(20,20));a=(a+a.T)/2
            np.fill_diagonal(a,1);names=[f'x{i}' for i in range(20)]
            cross=pd.DataFrame(a,index=names,columns=names)
            target=pd.Series(rng.uniform(size=20),index=names)
            if seed==3:
                cross.iloc[0,:]=0.;cross.iloc[:,0]=0.;target.iloc[0]=0.
            if seed==4: cross[:]=.5;target[:]=0.
            self.assertEqual(get_features_dyn(-1,cross,target),fast_native_ranking(cross,target))

    def test_missing_pairs_use_native_mi_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);matrix=directory/'x.npy';x=np.random.RandomState(1).normal(size=(40,3));np.save(matrix,x)
            h=[compute_mi(x[:,i],x[:,i],random_state=42)[0] for i in range(3)]
            job=dict(matrix=str(matrix),matrix_sha=run.file_hash(matrix),i=0,js=[1,2],seed=42,
                entropy_i=h[0],entropy_js=h[1:],path=str(directory/'row.npz'))
            path=cross_row(job);before=run.file_hash(path);self.assertEqual(cross_row(job),path)
            self.assertEqual(before,run.file_hash(path))
            with np.load(path) as z:
                expected=[compute_mi(x[:,0],x[:,j],random_state=42)[0]/(.5*(h[0]+h[j])) for j in (1,2)]
                np.testing.assert_array_equal(z['values'],expected)


if __name__=='__main__':unittest.main()
