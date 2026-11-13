from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_standalone_kan_screen import np,forward,transformations


class NetworkAuditChecks(unittest.TestCase):
    def test_constant_edge_pruning_preserves_bias_and_prediction(self):
        knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3]
        linear=np.array([knots[i+1:i+4].mean() for i in range(8)])
        coefficients=np.stack([np.full(8,2.),linear])[:,None,:]
        network=dict(target_mean=100.,target_std=10.,layers=[dict(name='one',
            coefficients=coefficients.tolist(),bias=[3.],hidden_coordinates=False)])
        x=np.random.RandomState(12).uniform(-.45,.45,(60,2))
        expected=100+10*(5+x[:,1])
        np.testing.assert_allclose(forward(network,x),expected,atol=1e-11)
        pruned,symbolic,summary=transformations(network,x)
        self.assertEqual(summary[0]['kept_edges'],1)
        self.assertEqual(summary[0]['symbolic_edges'],2)
        np.testing.assert_allclose(forward(pruned,x),expected,atol=1e-10)
        np.testing.assert_allclose(forward(symbolic,x),expected,atol=1e-10)

    def test_whole_network_symbolic_composition_uses_transformed_hidden_coordinates(self):
        knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3]
        linear=np.array([knots[i+1:i+4].mean() for i in range(8)])
        network=dict(target_mean=0.,target_std=1.,layers=[
            dict(name='a',coefficients=(2*linear)[None,None,:].tolist(),bias=[.1],hidden_coordinates=False),
            dict(name='b',coefficients=(3*linear)[None,None,:].tolist(),bias=[.2],hidden_coordinates=True)])
        x=np.linspace(-.4,.4,80)[:,None]
        expected=1.5*np.tanh(2*x[:,0]+.1)+.2
        np.testing.assert_allclose(forward(network,x),expected,atol=1e-12)
        _,symbolic,summary=transformations(network,x)
        self.assertTrue(all(s['symbolic_edges']==1 for s in summary))
        np.testing.assert_allclose(forward(symbolic,x),expected,atol=1e-10)


if __name__=='__main__':unittest.main()
