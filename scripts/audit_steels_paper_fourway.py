"""Independently verify each completed fold and the shared feature rankings."""
import argparse
import gzip
import json
from pathlib import Path

import audit_steels_shap_fold_v3 as original


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'benchmarks/steels-paper-fourway-20261106'
REFERENCE = ROOT / 'benchmarks/steels-paper-reproduction-20261102/lmp-v3-paper-fixed50'


def main(mode):
    dataset = (Path.home() / '.conda/envs/kan-cgcnn-cuda/Lib/site-packages'
               / 'matminer/datasets/matbench_steels.json.gz')
    with gzip.open(dataset, 'rt') as stream:
        data = json.load(stream)
    original.targets = {f'id{i}': row[1] for i, row in zip(data['index'], data['data'])}
    original.BASE = BASE
    protocol = original.read(BASE / mode / 'protocol.json')
    for filename, expected in protocol['sources'].items():
        assert original.sha(Path(filename)) == expected, filename
    for filename, expected in protocol['input_hashes'].items():
        assert original.sha(Path(filename)) == expected, filename
    reference_protocol = REFERENCE / mode / 'protocol.json'
    assert original.sha(reference_protocol) == protocol['baseline_reuse']['protocol_sha256']
    completed = list((BASE / mode).glob('*/fold-*/complete.json'))
    if not completed:
        raise RuntimeError('No completed comparison folds to audit.')
    for path in completed:
        folder = path.parent
        arm = folder.parent.name
        fold = int(folder.name.split('-')[1])
        assert arm in ('mm_modnet', 'mm_kan', 'mm_mv_kan')
        for ranking_path in folder.glob('inner-*/ranking.json'):
            ranking = original.read(ranking_path)
            if arm.endswith('_kan'):
                donor = Path(ranking['ranking_source'])
                assert original.sha(donor) == ranking['ranking_source_sha256']
                expected = original.read(donor)
                assert ranking['subsets'] == expected['subsets']
                assert ranking['train_ids'] == expected['train_ids']
            else:
                assert original.sha(ranking_path.parent / 'pilot/model.pkl') == ranking['pilot_sha256']
        for selection_path in folder.glob('*/inner-selection.json'):
            selection = original.read(selection_path)
            for member in selection['members']:
                fit = original.read(Path(member['directory']) / 'result.json')
                expected_variant = 'bspline' if arm.endswith('_kan') else 'modnet'
                assert fit['variant'] == expected_variant
                assert fit['seed'] == member['seed']
        record = original.audit(mode, arm, fold)
        (folder / 'independent-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(arm, fold, 'passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'formal'])
    main(parser.parse_args().mode)
