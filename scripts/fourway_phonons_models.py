"""Native MODNet versus a complete near-linear B-spline network."""
from run_modnet_official_matched import set_seed
from standalone_kan_models import SplineEdgeLayer, np, tf
from modnet.models import MODNetModel
from fourway_phonons_policy import dense_widths, dense_parameters, spline_widths, spline_parameters


def build_model(variant, config, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)
    n, width = config['n_feat'], config['width']
    if variant == 'modnet':
        widths = dense_widths(width)
        model = MODNetModel(targets=[[['target']]], weights={'target': 1},
            n_feat=n, num_neurons=[[w] for w in widths], act='elu').model
        expected = dense_parameters(n, widths)
    elif variant == 'kan':
        a, b = spline_widths(n, width)
        inputs = tf.keras.Input((n,))
        x = inputs
        for i, size in enumerate((a, b, 1)):
            x = SplineEdgeLayer(size, hidden_coordinates=i > 0,
                               seed=seed+310+i, name=f'kan_{i}')(x)
        model = tf.keras.Model(inputs, x)
        knots = np.r_[[-.5]*3, np.linspace(-.5, .5, 6), [.5]*3]
        greville = np.array([knots[i+1:i+4].mean() for i in range(8)])
        for i, layer in enumerate(l for l in model.layers if isinstance(l, SplineEdgeLayer)):
            nin, nout, _ = map(int, layer.coefficients.shape)
            rng = np.random.RandomState(seed+1900+i)
            limit = np.sqrt(6/(nin+nout))
            slopes = rng.uniform(-limit, limit, (nin, nout))
            coefficients = (2. if layer.hidden_coordinates else 1.) * slopes[..., None] * greville
            coefficients += rng.normal(0, .001/np.sqrt(nin), size=coefficients.shape)
            layer.coefficients.assign(coefficients.astype('float32'))
            layer.bias.assign(tf.zeros_like(layer.bias))
        expected = spline_parameters(n, a, b)
    else:
        raise ValueError(variant)
    if model.count_params() != expected:
        raise ValueError('Unexpected effective parameter count')
    return model
