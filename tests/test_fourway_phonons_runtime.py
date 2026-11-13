"""Real feature-fitting invariance, with intentionally adversarial holdouts."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_fourway_phonons import prepare_input,read,digest,file_hash,write_json,np,pd


class FourwayIsolationTests(unittest.TestCase):
    def test_holdout_changes_cannot_change_selected_features_or_training_transforms(self):
        rng=np.random.RandomState(719)
        ids=[f's{i}' for i in range(64)];columns=[f'f{i}' for i in range(32)]
        frame=pd.DataFrame(rng.normal(size=(64,32)),index=ids,columns=columns)
        frame['target']=3*frame.f0-frame.f1+rng.normal(size=64)*.1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol={'source_sha256':{}};write_json(root/'protocol.json',protocol)
            infos=[]
            for repeat in range(2):
                data=frame.copy()
                if repeat:
                    data.loc[ids[48:],'target']+=1e6
                    data.loc[ids[48:],columns]*=-1e4
                cache=root/f'data-{repeat}.pkl.gz';data.to_pickle(cache,compression='gzip',protocol=4)
                job=dict(directory=str(root/f'input-{repeat}'),cache=str(cache),cache_sha256=file_hash(cache),
                    train_ids=ids[:48],validation_ids=ids[48:],outer_test_ids=['unseen'],count=16,seed=719,
                    run_fingerprint=digest(protocol),protocol_path=str(root/'protocol.json'))
                infos.append(prepare_input(job))
            self.assertEqual(infos[0]['selected_features'],infos[1]['selected_features'])
            self.assertEqual(infos[0]['target_mean'],infos[1]['target_mean'])
            self.assertEqual(infos[0]['target_std'],infos[1]['target_std'])
            with np.load(root/'input-0/arrays.npz') as a,np.load(root/'input-1/arrays.npz') as b:
                np.testing.assert_array_equal(a['x_train'],b['x_train'])
                np.testing.assert_array_equal(a['y_train'],b['y_train'])
                self.assertFalse(np.array_equal(a['x_validation'],b['x_validation']))
                self.assertFalse(np.array_equal(a['y_validation'],b['y_validation']))
            bad=dict(job,directory=str(root/'leaked'),outer_test_ids=[ids[0]])
            with self.assertRaisesRegex(ValueError,'Outer contamination'):prepare_input(bad)
            # Cached numerical arrays must be authenticated, not blindly reused.
            with (root/'input-1/arrays.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed preprocessing artifact'):prepare_input(job)


if __name__=='__main__':unittest.main()
