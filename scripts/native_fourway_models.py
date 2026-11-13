"""Latest near-linear pure B-spline predictor inside the native MODNet trainer."""
from run_modnet_official_matched import np, tf
from standalone_kan_models import SplineEdgeLayer
from modnet.models import MODNetModel
from functools import lru_cache
import math


def dense_count(n, widths):
    sizes = [n] + list(widths) + [1]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


def spline_count(n, a, b):
    return 8 * (n * a + a * b + b) + a + b + 1


@lru_cache(None)
def matched_widths(n, widths):
    target = dense_count(n, widths)
    choices = []
    # The ratio is inherited from the latest 17->8->1 B-spline experiment.
    # Model size is chosen by parameter counts, never by validation scores.
    for a in range(1, max(513, int(math.sqrt(target)) + 2)):
        for b in range(max(1, round(a * 8 / 17) - 8), round(a * 8 / 17) + 9):
            error = abs(spline_count(n, a, b) / target - 1)
            if error <= .05:
                choices.append((abs(math.log(b / a / (8 / 17))), error, a, b))
    if not choices:
        raise ValueError(('No parameter-matched spline network', n, widths, target))
    _, _, a, b = min(choices)
    return a, b


class NativeSplineModel(MODNetModel):
    """Only build_model changes; fit/evaluate/scaling are the original methods."""
    def __init__(self, *args, spline_seed=0, input_scale='minmax', **kwargs):
        self.spline_seed = spline_seed
        self.input_scale = input_scale
        super().__init__(*args, **kwargs)

    def build_model(self, targets, n_feat, num_neurons, **kwargs):
        widths = tuple(int(x[0]) for x in num_neurons)
        a, b = matched_widths(n_feat, widths)
        inputs = tf.keras.Input((n_feat,))
        x = inputs
        knots = np.r_[[-.5] * 3, np.linspace(-.5, .5, 6), [.5] * 3]
        greville = np.array([knots[i+1:i+4].mean() for i in range(8)])
        for i, units in enumerate((a, b, 1)):
            # StandardScaler is retained in the GA. Bound those first-layer
            # coordinates smoothly, avoiding destructive clipping to the grid.
            bounded = i > 0 or self.input_scale == 'standard'
            layer = SplineEdgeLayer(units, hidden_coordinates=bounded,
                seed=self.spline_seed+310+i,
                name=targets[0][0][0] if i == 2 else f'kan_{i}')
            x = layer(x)
            nin, nout, _ = map(int, layer.coefficients.shape)
            rng = np.random.RandomState(self.spline_seed+1900+i)
            limit = np.sqrt(6/(nin+nout))
            slopes = rng.uniform(-limit, limit, (nin, nout))
            coef = (2. if bounded else 1.) * slopes[..., None] * greville
            coef += rng.normal(0, .001/np.sqrt(nin), size=coef.shape)
            layer.coefficients.assign(coef.astype('float32'))
            layer.bias.assign(tf.zeros_like(layer.bias))
        model = tf.keras.Model(inputs, x)
        assert model.count_params() == spline_count(n_feat, a, b)
        self.spline_widths = (a, b, 1)
        self.reference_dense_parameters = dense_count(n_feat, widths)
        return model
