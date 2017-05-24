from sympy.core.symbol import Symbol
from .input_size import input_size
from sklearn.linear_model.logistic import LogisticRegression
from pyearth.earth import Earth
from .base import call_method_or_dispatch

def syms_x(estimator):
    return [Symbol('x%d' % d) for d in range(input_size(estimator))]

def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

syms_dispatcher = {LogisticRegression: syms_x,
                   Earth: syms_earth}
syms = call_method_or_dispatch('syms', syms_dispatcher)
