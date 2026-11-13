"""Direct physical-coordinate KANs with exact additive/product decompositions.

Custom TensorFlow implementation inspired by MultKAN, not the pykan package.
Only column-wise train-fitted scaling is used; no mixing/LayerNorm of coordinates.
"""
from run_modnet_official_matched import set_seed
from physical_feature_modules import PHYSICAL_FEATURES, cubic_basis, WIDTHS
from modnet.models import MODNetModel
import numpy as np
import tensorflow as tf

PAIRS = ((0, 8), (0, 12), (8, 10), (4, 7))
SHORT_NAMES = ('mean_mass', 'mass_range', 'mass_deviation', 'max_mass',
               'mean_radius', 'radius_deviation', 'density', 'volume_per_atom',
               'mean_bond_length', 'bond_length_deviation', 'coordination',
               'neighbor_variation', 'mean_electronegativity', 'electronegativity_deviation',
               'mean_valence', 'valence_deviation')
VARIANTS = tuple(scope + '_' + kind for scope in ('physical', 'hybrid')
                 for kind in ('additive', 'product', 'mlp8', 'mlp12'))


def basis_numpy(x):
    """Independent numpy implementation for interpretation and numerical checks."""
    x = np.asarray(x, dtype=np.float64)
    knots = np.r_[np.repeat(-.5, 3), np.linspace(-.5, .5, 6), np.repeat(.5, 3)]
    z = np.clip(x, -.5, .5-1e-6)[..., None]
    b = ((z >= knots[:-1]) & (z < knots[1:])).astype(float)
    for k in range(1, 4):
        n = 11-k
        ld, rd = knots[k:k+n]-knots[:n], knots[k+1:k+1+n]-knots[1:1+n]
        left = np.divide(z-knots[:n], ld, out=np.zeros(z.shape[:-1]+(n,)), where=ld != 0)
        right = np.divide(knots[k+1:k+1+n]-z, rd, out=np.zeros(z.shape[:-1]+(n,)), where=rd != 0)
        b = left*b[..., :n] + right*b[..., 1:n+1]
    return np.concatenate((b, x[..., None]), axis=-1)


def statistics(x_train):
    b = basis_numpy(x_train)
    center = b.mean(axis=0)
    centered = b-center
    products = np.stack([np.einsum('ni,nj->ij', centered[:, i], centered[:, j])/len(b)
                         for i, j in PAIRS])
    return center, products


@tf.keras.utils.register_keras_serializable(package='InterpretablePhonons')
class PhysicalResponse(tf.keras.layers.Layer):
    def __init__(self, indices, kind, centers, product_centers, seed=0, hybrid=False, **kwargs):
        super().__init__(**kwargs)
        self.indices, self.kind, self.seed = list(indices), kind, int(seed)
        self.centers, self.product_centers = np.asarray(centers).tolist(), np.asarray(product_centers).tolist()
        self.hybrid = bool(hybrid)
        if len(indices) != 16 or kind not in ('additive', 'product', 'mlp8', 'mlp12'):
            raise ValueError('Invalid physical response')
        if kind.startswith('mlp'):
            width = int(kind[3:])
            self.mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(width, activation='elu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1)),
                tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+2))])

    def build(self, shape):
        if self.kind in ('additive', 'product'):
            self.main = self.add_weight('main', (16, 9), initializer=tf.keras.initializers.RandomNormal(stddev=.05, seed=self.seed))
            self.bias = self.add_weight('bias', (1,), initializer='zeros')
            if self.kind == 'product':
                self.factors = self.add_weight('factors', (4, 2, 9),
                    initializer=tf.keras.initializers.RandomNormal(stddev=.15, seed=self.seed+3))
        if self.hybrid:
            self.gate = self.add_weight('gate', (), initializer='zeros')
        super().build(shape)

    def call(self, inputs):
        x = tf.gather(inputs, self.indices, axis=-1)
        if self.kind.startswith('mlp'):
            out = self.mlp(x)
        else:
            b = tf.concat([cubic_basis(x), x[..., None]], axis=-1) - tf.cast(self.centers, x.dtype)
            main = tf.einsum('nib,ib->ni', b, self.main)
            out = tf.reduce_sum(main, axis=-1, keepdims=True)+self.bias
            if self.kind == 'product':
                for p, (i, j) in enumerate(PAIRS):
                    left = tf.einsum('nb,b->n', b[:, i], self.factors[p, 0])
                    right = tf.einsum('nb,b->n', b[:, j], self.factors[p, 1])
                    center = tf.einsum('i,ij,j->', self.factors[p, 0],
                                      tf.cast(self.product_centers[p], x.dtype), self.factors[p, 1])
                    out += (left*right-center)[:, None]
        return out*self.gate if self.hybrid else out

    def get_config(self):
        return dict(super().get_config(), indices=self.indices, kind=self.kind,
                    centers=self.centers, product_centers=self.product_centers,
                    seed=self.seed, hybrid=self.hybrid)

    def export(self):
        if self.kind.startswith('mlp'):
            return None
        return dict(kind=self.kind, main=self.main.numpy().tolist(), bias=float(self.bias.numpy()[0]),
                    factors=self.factors.numpy().tolist() if self.kind == 'product' else None,
                    gate=float(self.gate.numpy()) if self.hybrid else 1., centers=self.centers,
                    product_centers=self.product_centers, indices=self.indices,
                    features=PHYSICAL_FEATURES, short_names=SHORT_NAMES, pairs=PAIRS)


def decompose(x_physical, exported):
    b = basis_numpy(x_physical)-np.asarray(exported['centers'])
    main = np.einsum('nib,ib->ni', b, exported['main'])
    terms = [main]
    if exported['factors'] is not None:
        f = np.asarray(exported['factors'])
        products = []
        for p, (i, j) in enumerate(PAIRS):
            center = f[p, 0] @ np.asarray(exported['product_centers'][p]) @ f[p, 1]
            products.append((b[:, i] @ f[p, 0]) * (b[:, j] @ f[p, 1])-center)
        terms.append(np.column_stack(products))
    terms = np.concatenate(terms, axis=1)*exported['gate']
    return terms, exported['bias']*exported['gate']


def expected_parameters(variant, n_features=256):
    scope, kind = variant.split('_')
    response = 145 if kind in ('additive', 'mlp8') else 217
    trunk = (n_features+1)*96+(96+1)*72+(72+1)*36+(36+1)*36+(36+1)
    return response if scope == 'physical' else trunk+response+1


def build_model(variant, features, x_train, seed):
    if variant not in VARIANTS or any('SISSO' in f.upper() for f in features):
        raise ValueError('Invalid variant/features')
    tf.keras.backend.clear_session()
    set_seed(seed)
    indices = [features.index(f) for f in PHYSICAL_FEATURES]
    center, product_center = statistics(x_train[:, indices])
    scope, kind = variant.split('_')
    # Construct every MODNet trunk first with the same seed. Branch initializers
    # have separate explicit seeds; hybrid initial predictions exactly match it.
    original = MODNetModel(targets=[[['target']]], weights={'target': 1}, n_feat=len(features),
                          num_neurons=WIDTHS, act='elu').model
    inputs = original.input if scope == 'hybrid' else tf.keras.Input((len(features),))
    branch = PhysicalResponse(indices, kind, center, product_center, seed=seed+800,
                              hybrid=scope == 'hybrid', name='physical_response')(inputs)
    output = tf.keras.layers.Add()([original.outputs[0], branch]) if scope == 'hybrid' else branch
    model = tf.keras.Model(inputs, output)
    if model.count_params() != expected_parameters(variant, len(features)):
        raise ValueError('Parameter count mismatch')
    return model
