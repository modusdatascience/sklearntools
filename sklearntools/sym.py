from pyearth.earth import Earth
from pyearth.export import export_sympy_term_expressions, export_sympy
from sklearn.linear_model.logistic import LogisticRegression
from sympy.core.numbers import RealNumber
from sympy.functions.elementary.exponential import exp
from sympy.core.symbol import Symbol
from sklearn.linear_model.base import LinearRegression
import numpy as np
from sympy.printing.lambdarepr import NumPyPrinter
from sympy.printing.jscode import JavascriptCodePrinter
import os
from resources import resources
from mako.template import Template

def call_method_or_dispatch(method_name, dispatcher):
    def _call_method_or_dispatch(estimator, *args, **kwargs):
        try:
            return getattr(estimator, method_name)(*args, **kwargs)
        except AttributeError:
            for klass in type(estimator).mro():
                if klass in dispatcher:
                    exporter = dispatcher[klass]
                    return exporter(estimator, *args, **kwargs)
            raise
    _call_method_or_dispatch.__name__ = method_name
    return _call_method_or_dispatch

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

def syms_x(estimator):
    return [Symbol('x%d' % d) for d in range(input_size(estimator))]

def syms_earth(estimator):
    return [Symbol(label) for label in estimator.xlabels_]

syms_dispatcher = {LogisticRegression: syms_x,
                   Earth: syms_earth}
syms = call_method_or_dispatch('syms', syms_dispatcher)

def sym_predict_linear(estimator):
    if hasattr(estimator, 'intercept_'):
        expression = RealNumber(estimator.intercept_[0])
    else:
        expression = RealNumber(0)
    symbols = syms(estimator)
    for coef, sym in zip(np.ravel(estimator.coef_), symbols):
        expression += RealNumber(coef) * sym
    return expression

def sym_predict_logist_regression(logistic_regression):
    return RealNumber(1) / (RealNumber(1) + exp(-sym_predict_linear(logistic_regression)))
    
sym_predict_dispatcher = {Earth: export_sympy,
                     LogisticRegression: sym_predict_logist_regression,
                     LinearRegression: sym_predict_linear}
sym_predict = call_method_or_dispatch('sym_predict', sym_predict_dispatcher)

sym_predict_proba_dispatcher = {LogisticRegression: sym_predict_logist_regression}
sym_predict_proba = call_method_or_dispatch('sym_predict_proba', sym_predict_proba_dispatcher)

sym_transform_dispatcher = {Earth: export_sympy_term_expressions}
sym_transform = call_method_or_dispatch('sym_transform', sym_transform_dispatcher)

def fallback(*args):
    def _fallback(*inner_args, **kwargs):
        steps = list(args)
        while steps:
            try:
                return steps.pop(0)(*inner_args, **kwargs)
            except AttributeError:
                if not steps:
                    raise
    _fallback.__name__ = args[0].__name__
    return _fallback

sym_update_dispatcher = {}
sym_update = fallback(call_method_or_dispatch('sym_update', sym_update_dispatcher), sym_transform)

class STNumpyPrinter(NumPyPrinter):
    def _print_Max(self, expr):
        return 'maximum(' + ','.join(self._print(i) for i in expr.args) + ')'

    def _print_NaNProtect(self, expr):
        return 'where(isnan(' + ','.join(self._print(a) for a in expr.args) + '), 0, ' \
            + ','.join(self._print(a) for a in expr.args) + ')'

    def _print_Missing(self, expr):
        return 'isnan(' + ','.join(self._print(a) for a in expr.args) + ').astype(float)'

class STJavaScriptPrinter(JavascriptCodePrinter):
    def _print_Max(self, expr):
        return 'Math.max(' + ','.join(self._print(i) for i in expr.args) + ')'
 
    def _print_NaNProtect(self, expr):
        return 'nanprotect(' + ','.join(self._print(i) for i in expr.args) + ')'
 
    def _print_Missing(self, expr):
        return 'missing(' + ','.join(self._print(a) for a in expr.args) + ')'
    
javascript_template_filename = os.path.join(resources, 'template.mako.js')
with open(javascript_template_filename) as infile:
    javascript_template = Template(infile.read())

def javascript_str(function_name, estimator, method=sym_predict):
    input_names = [sym.name for sym in syms(estimator)]
    expression = method(estimator)
    return javascript_template.render(function_name=function_name, input_names=input_names,
                                      function_code=STJavaScriptPrinter().doprint(expression))

