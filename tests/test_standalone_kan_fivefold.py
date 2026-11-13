from pathlib import Path
import sys
import json
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from standalone_kan_fivefold_policy import official_ids,validate_fivefold,metrics


class FrozenFiveFoldTests(unittest.TestCase):
    def test_official_position_mapping_and_disjoint_coverage(self):
        root=Path(__file__).resolve().parents[1]
        folds=[]
        for i in range(5):
            p=root/f'benchmarks/official-aligned-20261021/matbench_phonons/fold-{i}/outer-split.json'
            raw=json.loads(p.read_text(encoding='utf-8-sig'));train,test=official_ids(raw)
            folds.append(dict(train_ids=train,test_ids=test))
            raw['official_test_ids']=list(reversed(raw['official_test_ids']))
            with self.assertRaises(ValueError):official_ids(raw)
        validate_fivefold(folds)
        with self.assertRaises(ValueError):validate_fivefold(folds[:4]+[folds[0]])

    def test_aggregate_counts_losing_fold_and_uses_ratio_of_means(self):
        result=metrics([9,18,27,36,60],[10,20,30,40,50])
        self.assertEqual(result['wins'],4)
        self.assertEqual(result['kan_mean'],30)
        self.assertEqual(result['mlp_mean'],30)
        self.assertEqual(result['relative_improvement'],0)
        self.assertEqual(result['fold_absolute_improvement'],[1,2,3,4,-10])
        with self.assertRaises(ValueError):metrics([1]*4,[2]*4)


if __name__=='__main__':unittest.main()
