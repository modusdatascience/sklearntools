from sympy.core.numbers import RealNumber
from .syms import syms
import numpy as np
from sympy.functions.elementary.exponential import exp
from pyearth.earth import Earth
from pyearth.export import export_sympy
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.base import LinearRegression
from .base import call_method_or_dispatch
from sklearn.tree.tree import DecisionTreeRegressor
from sklearntools.sym.adapters.trees import sym_predict_decision_tree_regressor

def sym_predict_linear(estimator):
    if hasattr(estimator, 'intercept_'):
        expression = RealNumber(estimator.intercept_[0])
    else:
        expression = RealNumber(0)
    symbols = syms(estimator)
    for coef, sym in zip(np.ravel(estimator.coef_), symbols):
        expression += RealNumber(coef) * sym
    return expression

def sym_predict_logistic_regression(logistic_regression):
    return RealNumber(1) / (RealNumber(1) + exp(-sym_predict_linear(logistic_regression)))

sym_predict_dispatcher = {Earth: export_sympy,
                          LogisticRegression: sym_predict_logistic_regression,
                          LinearRegression: sym_predict_linear,
                          DecisionTreeRegressor: sym_predict_decision_tree_regressor}

sym_predict = call_method_or_dispatch('sym_predict', sym_predict_dispatcher)
