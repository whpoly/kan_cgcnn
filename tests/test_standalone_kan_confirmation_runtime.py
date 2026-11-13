"""Run in the pinned training environment; tests the evaluation-label boundary."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_standalone_kan_confirmation as study
from run_modnet_official_matched import digest,file_hash,write_json
import pickle
import numpy as np


class FixedFitBoundaryTests(unittest.TestCase):
    def test_fixed_training_ignores_evaluation_labels_and_has_no_validation_callback(self):
        root=study.ROOT
        inp=study.read(study.OLD/'protocol.json')['parent_protocol']['source_inputs'][0]
        with tempfile.TemporaryDirectory(prefix='kan-confirm-smoke-',dir=str(root/'benchmarks')) as temporary:
            base=Path(temporary)
            self.assertEqual(base.parent.resolve(),(root/'benchmarks').resolve())
            fitted=[]
            original=study.tf.keras.Model.fit
            def spy(model,*args,**kwargs):
                self.assertNotIn('validation_data',kwargs)
                self.assertNotIn('validation_split',kwargs)
                self.assertEqual(kwargs['epochs'],2)
                self.assertFalse(any(isinstance(c,study.tf.keras.callbacks.EarlyStopping) for c in kwargs.get('callbacks',[])))
                fitted.append(True)
                return original(model,*args,**kwargs)
            jobs=[]
            for bank in ('enhanced','physical'):
                for variant in ('kan','matched_mlp'):
                    jobs.append(dict(directory=str(base/(bank+'-'+variant)),phase='confirmation',bank=bank,variant=variant,
                        repeat=0,inner=0,seed=580281,lr=.001,input=inp,run_fingerprint='smoke',selection_sha256='smoke',
                        fixed_epochs=2,max_epochs=400,expected_parameters=study.COUNTS[bank][variant]))
            with patch.object(study.tf.keras.Model,'fit',spy):
                for j in jobs:
                    result=study.fixed_fit(j)
                    self.assertEqual(result['epochs'],2)
                    self.assertEqual(study.fixed_fit(j)['signature'],result['signature'])
                # Alter only held-out labels: model weights must remain unchanged.
                changed=base/'changed-input';changed.mkdir()
                info=dict(inp['source'])
                with np.load(Path(info['directory'])/'arrays.npz') as a:arrays={k:a[k] for k in a.files}
                arrays['y_validation']=arrays['y_validation']+100000.
                np.savez_compressed(changed/'arrays.npz',**arrays)
                info['directory']=str(changed);info['artifacts']={'arrays.npz':file_hash(changed/'arrays.npz')}
                write_json(changed/'manifest.json',info)
                inp2=study.as_input(info) if 'target_mean' in info else dict(inp,source=info,
                    source_manifest=str(changed/'manifest.json'),source_manifest_sha256=file_hash(changed/'manifest.json'))
                prior=jobs[2];j=dict(prior,directory=str(base/'changed-labels'),input=inp2)
                result=study.fixed_fit(j)
            self.assertEqual(len(fitted),5)
            with (Path(prior['directory'])/'model.pkl').open('rb') as f:a=pickle.load(f)
            with (Path(j['directory'])/'model.pkl').open('rb') as f:b=pickle.load(f)
            for wa,wb in zip(a['weights'],b['weights']):np.testing.assert_array_equal(wa,wb)


if __name__=='__main__':unittest.main()
