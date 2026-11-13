"""Controls against weak-baseline, unequal-budget and attribution mistakes."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fourway_phonons_policy import (candidate_grid,dense_widths,dense_parameters,
    spline_widths,spline_parameters,choose,baseline_gate,factorial_effects)


class FourwayPolicyTests(unittest.TestCase):
    def test_complete_symmetric_budget_and_capacity(self):
        grid=candidate_grid()
        self.assertEqual(len(grid),192)
        self.assertEqual(len({tuple(c.values()) for c in grid}),192)
        self.assertEqual(len(grid)*5*5*4,19200)
        for n,w in {(c['n_feat'],c['width']) for c in grid+candidate_grid(True)}:
            a,b=spline_widths(n,w)
            self.assertLessEqual(abs(spline_parameters(n,a,b)/dense_parameters(n,dense_widths(w))-1),.05)
        self.assertEqual(spline_widths(256,96),[17,8])

    def test_selection_requires_every_candidate_and_fold(self):
        grid=candidate_grid(True)
        rows=[dict(candidate=c['candidate'],inner=i,validation_mae=10+c['candidate']-i,
                   directory=f"c{c['candidate']}i{i}") for c in grid for i in range(2)]
        result=choose(rows,grid,2,1)
        self.assertEqual(result['selected_candidates'],[0])
        self.assertEqual(result['members'],['c0i0','c0i1'])
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[dict(rows[-1],validation_mae=float('nan'))]):
            with self.assertRaises(ValueError):choose(bad,grid,2,1)

    def test_baseline_gate_rejects_weak_or_partial_reference(self):
        self.assertTrue(baseline_gate([34.013]*5,34.2751)['passed'])
        self.assertFalse(baseline_gate([42.467]*5,34.2751)['passed'])
        self.assertTrue(baseline_gate([30]*5,34.2751)['passed'])
        with self.assertRaises(ValueError):baseline_gate([34]*4,34.2751)
        self.assertFalse(baseline_gate([99],34.2751,True)['establishes_reproduction'])

    def test_interaction_is_difference_of_kan_effects(self):
        e=factorial_effects(dict(mm_modnet=40,mm_mv_modnet=34,mm_kan=38,mm_mv_kan=29))
        self.assertEqual(e['mattervial_with_modnet'],6)
        self.assertEqual(e['kan_with_matminer'],2)
        self.assertEqual(e['kan_with_mattervial'],5)
        self.assertEqual(e['interaction'],3)
        self.assertEqual(e['total'],11)
        self.assertEqual(e['total'],e['mattervial_with_modnet']+e['kan_with_mattervial'])


if __name__=='__main__':unittest.main()
