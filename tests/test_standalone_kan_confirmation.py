"""Leakage-boundary and predeclared-decision tests (no TensorFlow required)."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from standalone_kan_confirmation_policy import make_splits,choose,evaluate_gate,BANKS,VARIANTS,LRS


class ConfirmationPolicyTests(unittest.TestCase):
    def test_confirmation_holdout_never_enters_tuning_partitions(self):
        ids=['id'+str(i) for i in range(1012)]
        splits=make_splits(ids)
        held=[]
        for s in splits:
            a,b,h=map(set,(s['tuning_train_ids'],s['tuning_validation_ids'],s['validation_ids']))
            self.assertFalse(a&b or a&h or b&h)
            self.assertEqual(a|b,set(s['train_ids']))
            self.assertEqual(a|b|h,set(ids))
            held+=s['validation_ids']
        self.assertEqual(len(held),len(set(held)))
        self.assertEqual(set(held),set(ids))
        self.assertEqual(splits,make_splits(list(reversed(ids))))

    def test_equal_tuning_budget_required_and_fixed_epochs_come_from_selected_lr(self):
        records=[dict(lr=lr,repeat=s,inner=0,validation_mae=10 if lr==.001 else 12,
                      best_epoch=80+3*s if lr==.001 else 10) for lr in LRS for s in range(2)]
        choice=choose(records,2)
        self.assertEqual(choice['lr'],.001)
        self.assertEqual(choice['epochs'],82)
        self.assertEqual(choice,choose(list(reversed(records)),2))
        with self.assertRaises(ValueError):choose(records[:-1],2)

    def test_primary_failure_cannot_be_rescued_by_secondary(self):
        records=[dict(bank=b,variant=v,repeat=s,inner=i,
                      validation_mae=100 if v=='matched_mlp' else (99 if b=='enhanced' else 50))
                 for b in BANKS for v in VARIANTS for s in range(2) for i in range(5)]
        result=evaluate_gate(records)
        self.assertFalse(result['passed'])
        self.assertTrue(result['rows']['physical']['passed'])
        for r in records:
            if r['bank']=='enhanced' and r['variant']=='kan':r['validation_mae']=95
        self.assertTrue(evaluate_gate(records)['passed'])
        with self.assertRaises(ValueError):evaluate_gate(records[:-1])


if __name__=='__main__':unittest.main()
