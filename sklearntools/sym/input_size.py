from sklearn.linear_model.logistic import LogisticRegression
from pyearth.earth import Earth
from .base import call_method_or_dispatch

def input_size_from_coef(estimator):
    assert hasattr(estimator, 'coef_')
    coef = estimator.coef_
    n_inputs = coef.shape[-1]
    return n_inputs

def input_size_earth(estimator):
    return len(estimator.xlabels_)

input_size_dispatcher = {LogisticRegression: input_size_from_coef,
                         Earth: input_size_earth}
input_size = call_method_or_dispatch('input_size', input_size_dispatcher)
