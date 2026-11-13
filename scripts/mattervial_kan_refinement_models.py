"""Small KAN accuracy ablations with exact, active MLP parameter matches."""
from __future__ import annotations

from run_modnet_official_matched import set_seed
from physical_feature_modules import GroupFeatures, PHYSICAL_FEATURES, WIDTHS, cubic_basis
from modnet.models import MODNetModel
import tensorflow as tf

VARIANTS = ('modnet', 'group_mlp', 'group_kan', 'group_sparse_diagnostic', 'latent_mlp', 'latent_kan')
KAN_CANDIDATES = ('group_kan', 'latent_kan')
MLP_CANDIDATES = ('group_mlp', 'latent_mlp')
PAIRS = dict(group_kan='group_mlp', latent_kan='latent_mlp')


@tf.keras.utils.register_keras_serializable(package='MatterVialRefinement')
class ScalarSpline(tf.keras.layers.Layer):
    """16 inputs -> scalar: 128 cubic coefficients + 16 slopes + 1 bias."""
    def __init__(self, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(seed)

    def build(self, shape):
        if int(shape[-1]) != 16:
            raise ValueError('The capacity-matched scalar spline requires 16 inputs.')
        self.coefficients = self.add_weight('coefficients', (16,8),
            initializer=tf.keras.initializers.RandomNormal(stddev=.05,seed=self.seed))
        self.base = self.add_weight('base', (16,),
            initializer=tf.keras.initializers.RandomNormal(stddev=.05,seed=self.seed+1))
        self.bias = self.add_weight('bias', (1,), initializer='zeros')
        super().build(shape)

    def call(self, inputs):
        spline = tf.einsum('nib,ib->n',cubic_basis(inputs),self.coefficients)
        linear = tf.einsum('ni,i->n',inputs,self.base)
        return (spline+linear)[:,None]+self.bias

    def get_config(self):
        return dict(super().get_config(),seed=self.seed)


@tf.keras.utils.register_keras_serializable(package='MatterVialRefinement')
class LatentCorrection(tf.keras.layers.Layer):
    """36 -> 16 learned projection, bounded coordinates, zero-gated correction.

    Both branches have 145 core parameters. Including the shared projection
    (592) and gate (1), both corrections have exactly 738 active parameters.
    No sample statistics or labels are fitted by LayerNormalization.
    """
    def __init__(self, kind, seed=0, **kwargs):
        super().__init__(**kwargs)
        if kind not in ('mlp','kan'):
            raise ValueError(kind)
        self.kind, self.seed = kind, int(seed)
        self.projection = tf.keras.layers.Dense(16,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+71),name='projection')
        self.norm = tf.keras.layers.LayerNormalization(center=False,scale=False,epsilon=1e-5,name='coordinate_norm')
        if kind == 'kan':
            self.correction = ScalarSpline(seed=self.seed+72,name='scalar_kan')
        else:
            self.correction = tf.keras.Sequential([
                tf.keras.layers.Dense(8,activation='elu',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+72)),
                tf.keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed+73))],name='scalar_mlp')
        self.gate = self.add_weight('gate',shape=(),initializer='zeros')

    def call(self, inputs):
        coordinates = .5*tf.math.tanh(self.norm(self.projection(inputs)))
        return self.gate*self.correction(coordinates)

    def get_config(self):
        return dict(super().get_config(),kind=self.kind,seed=self.seed)


def build_model(variant, features, seed):
    if variant not in VARIANTS or any('SISSO' in f.upper() for f in features):
        raise ValueError('Invalid model or SISSO input.')
    tf.keras.backend.clear_session()
    set_seed(seed)
    original = MODNetModel(targets=[[['target']]],weights={'target':1},n_feat=len(features),
                          num_neurons=WIDTHS,act='elu').model
    if variant == 'modnet':
        return original
    dense = [l for l in original.layers if isinstance(l,tf.keras.layers.Dense)]
    if variant.startswith('latent_'):
        correction = LatentCorrection(variant.split('_')[1],seed=seed,name='latent_correction')(dense[-2].output)
        output = tf.keras.layers.Add(name='corrected_prediction')([original.outputs[0],correction])
        return tf.keras.Model(original.input,output)
    inputs = tf.keras.Input((len(features),),name='shared_features')
    kind = 'group_mlp' if variant == 'group_mlp' else 'group_sparse_kan'
    # y is normalized with TRAIN statistics. target_scale=1 gives the same
    # relative sparse/smooth objective as the original raw-target formulation.
    sparse = .01 if variant == 'group_sparse_diagnostic' else 0.
    smooth = .001 if variant == 'group_sparse_diagnostic' else 0.
    module = GroupFeatures([features.index(c) for c in PHYSICAL_FEATURES],kind,
        target_scale=1.,sparse_strength=sparse,smooth_strength=smooth,name='physical_module')
    x = tf.keras.layers.Concatenate()([inputs,module(inputs)])
    for index, old in enumerate(dense):
        new = tf.keras.layers.Dense.from_config(old.get_config())
        x = new(x)
        if index == 0:
            kernel,bias = old.get_weights()
            extended = tf.keras.initializers.GlorotUniform(seed=seed+109)(new.kernel.shape).numpy()
            extended[:len(features)] = kernel
            new.set_weights([extended,bias])
        else:
            new.set_weights(old.get_weights())
    return tf.keras.Model(inputs,x)
