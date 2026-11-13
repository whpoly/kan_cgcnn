"""Standalone spline-edge networks, with no MODNet trunk or bypass path."""
from run_modnet_official_matched import set_seed
from physical_feature_modules import cubic_basis, PHYSICAL_FEATURES
from modnet.models import MODNetModel
import numpy as np
import tensorflow as tf

CONFIGS = [('enhanced',v) for v in ('modnet','matched_mlp','kan','sparse_kan')] + [
    ('physical',v) for v in ('matched_mlp','kan','sparse_kan')]
COUNTS = {'enhanced':dict(modnet=35653,matched_mlp=35995,kan=35994,sparse_kan=35994),
          'physical':dict(matched_mlp=3355,kan=3354,sparse_kan=3354)}


@tf.keras.utils.register_keras_serializable(package='StandaloneKAN')
class SplineEdgeLayer(tf.keras.layers.Layer):
    def __init__(self,units,hidden_coordinates=False,sparse_strength=0.,smooth_strength=0.,seed=0,**kwargs):
        super().__init__(**kwargs)
        self.units=int(units);self.hidden_coordinates=bool(hidden_coordinates)
        self.sparse_strength=float(sparse_strength);self.smooth_strength=float(smooth_strength);self.seed=int(seed)

    def build(self,shape):
        width=int(shape[-1])
        self.coefficients=self.add_weight('coefficients',(width,self.units,8),
            initializer=tf.keras.initializers.RandomNormal(stddev=1/np.sqrt(width),seed=self.seed))
        self.coefficients.assign(self.coefficients-tf.reduce_mean(self.coefficients,axis=-1,keepdims=True))
        self.bias=self.add_weight('bias',(self.units,),initializer='zeros')
        super().build(shape)

    def call(self,inputs):
        x=.5*tf.math.tanh(inputs) if self.hidden_coordinates else inputs
        response=tf.einsum('nik,iok->no',cubic_basis(x),self.coefficients)+self.bias
        if self.sparse_strength:
            centered=self.coefficients-tf.reduce_mean(self.coefficients,axis=-1,keepdims=True)
            norms=tf.sqrt(tf.reduce_mean(tf.square(centered),axis=-1)+1e-10)
            self.add_loss(self.sparse_strength*tf.reduce_sum(norms)/self.units)
        if self.smooth_strength:
            differences=self.coefficients[:,:,2:]-2*self.coefficients[:,:,1:-1]+self.coefficients[:,:,:-2]
            self.add_loss(self.smooth_strength*tf.reduce_mean(tf.square(differences)))
        return response

    def get_config(self):
        return dict(super().get_config(),units=self.units,hidden_coordinates=self.hidden_coordinates,
                    sparse_strength=self.sparse_strength,smooth_strength=self.smooth_strength,seed=self.seed)


def build_model(bank,variant,seed):
    tf.keras.backend.clear_session();set_seed(seed)
    n=256 if bank=='enhanced' else 16
    if variant=='modnet':
        model=MODNetModel(targets=[[['target']]],weights={'target':1},n_feat=n,
                         num_neurons=[[96],[72],[36],[36]],act='elu').model
    elif variant=='matched_mlp':
        inputs=tf.keras.Input((n,));x=inputs
        widths=[96,72,36,45] if bank=='enhanced' else [75,27]
        for i,width in enumerate(widths):
            x=tf.keras.layers.Dense(width,activation='elu',name=f'mlp_{i}')(x)
        model=tf.keras.Model(inputs,tf.keras.layers.Dense(1,name='mlp_output')(x))
    else:
        if variant not in ('kan','sparse_kan'):raise ValueError(variant)
        inputs=tf.keras.Input((n,));x=inputs
        for i,width in enumerate((17,8,1)):
            x=SplineEdgeLayer(width,hidden_coordinates=i>0,
                sparse_strength=.001 if variant=='sparse_kan' else 0.,
                smooth_strength=.001 if variant=='sparse_kan' else 0.,
                seed=seed+310+i,name=f'kan_{i}')(x)
        model=tf.keras.Model(inputs,x)
    if model.count_params()!=COUNTS[bank][variant]:raise ValueError('Capacity mismatch')
    return model


def export_network(model,features,x_train):
    layers=[];values=np.asarray(x_train,dtype='float32');diagnostics=[]
    for layer in model.layers:
        if not isinstance(layer,SplineEdgeLayer):continue
        coords=.5*np.tanh(values) if layer.hidden_coordinates else values
        diagnostics.append(dict(name=layer.name,coordinate_std=np.std(coords,axis=0).tolist(),
            saturated_fraction=float(np.mean(abs(coords)>.49)),input_dimension=coords.shape[1]))
        layers.append(dict(name=layer.name,coefficients=layer.coefficients.numpy().tolist(),
            bias=layer.bias.numpy().tolist(),hidden_coordinates=layer.hidden_coordinates))
        values=layer(values).numpy()
    return dict(features=features,layers=layers,diagnostics=diagnostics,
        basis='8 clamped cubic B splines, 5 uniform intervals on [-0.5,0.5]',
        hidden_transform='0.5*tanh applied separately per hidden coordinate; no mixing normalization',
        interpretation='Only first-layer coordinates name input descriptors; deeper coordinates are learned functions')
