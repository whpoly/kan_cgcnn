"""Two-logit B-spline classifier; no regression target standardization.

Same integer-width candidate set and matching priority as the approved
regression policy, with parameter counts recomputed for two output classes.
"""
import math
from functools import lru_cache
from native_fourway_models import NativeSplineModel, SplineEdgeLayer, np, tf

def dense_count(n,widths):
    sizes=[n]+list(widths)+[2]
    return sum((a+1)*b for a,b in zip(sizes,sizes[1:]))

def spline_count(n,a,b):return 8*(n*a+a*b+2*b)+a+b+2

@lru_cache(None)
def mapping(n,widths):
    target=dense_count(n,widths);feasible=[];closest=None
    for a in range(1,max(513,int(math.sqrt(target))+2)):
        for b in range(max(1,round(a*8/17)-8),round(a*8/17)+9):
            error=abs(spline_count(n,a,b)/target-1);ratio=abs(math.log(b/a/(8/17)))
            if error<=.05:feasible.append((ratio,error,a,b))
            item=(error,ratio,a,b)
            if closest is None or item<closest:closest=item
    if feasible:_,_,a,b=min(feasible)
    else:_,_,a,b=closest
    return dict(widths=[a,b,2],reference_parameters=target,spline_parameters=spline_count(n,a,b),
        fallback_used=not bool(feasible),signed_relative_difference=spline_count(n,a,b)/target-1,
        rule='legacy_5pct' if feasible else 'closest_integer_network_either_direction')

class NativeClassificationSpline(NativeSplineModel):
    def build_model(self,targets,n_feat,num_neurons,**kwargs):
        assert list(self.num_classes.values())==[2] and not self.multi_label
        widths=tuple(int(x[0]) for x in num_neurons);m=mapping(n_feat,widths)
        inputs=tf.keras.Input((n_feat,));x=inputs
        knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3]
        greville=np.array([knots[i+1:i+4].mean() for i in range(8)])
        for i,units in enumerate(m['widths']):
            bounded=i>0 or self.input_scale=='standard'
            layer=SplineEdgeLayer(units,hidden_coordinates=bounded,seed=self.spline_seed+310+i,name=f'kan_classifier_{i}')
            x=layer(x);nin,nout,_=map(int,layer.coefficients.shape)
            rng=np.random.RandomState(self.spline_seed+1900+i);limit=np.sqrt(6/(nin+nout))
            slopes=rng.uniform(-limit,limit,(nin,nout))
            coefficients=(2. if bounded else 1.)*slopes[...,None]*greville
            coefficients+=rng.normal(0,.001/np.sqrt(nin),size=coefficients.shape)
            layer.coefficients.assign(coefficients.astype('float32'));layer.bias.assign(tf.zeros_like(layer.bias))
        probabilities=tf.keras.layers.Softmax(name=targets[0][0][0])(x)
        model=tf.keras.Model(inputs,probabilities);assert model.count_params()==m['spline_parameters']
        self.parameter_mapping=m;self.spline_widths=tuple(m['widths']);self.reference_dense_parameters=m['reference_parameters']
        return model

def probabilities(model,frame):
    cols=model.optimal_descriptors[:model.n_feat]
    x=np.nan_to_num(frame[cols].values)
    x=np.nan_to_num(model._scaler.transform(x),nan=-1)
    values=np.asarray(model.model(x,training=False))
    assert values.shape==(len(x),2) and np.isfinite(values).all()
    assert (values>=0).all() and (values<=1).all()
    np.testing.assert_allclose(values.sum(axis=1),1,atol=2e-6)
    return values
