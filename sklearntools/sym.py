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
from sympy.printing.python import PythonPrinter
import autopep8
from _collections import defaultdict
from itertools import chain
from operator import add

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

def sym_predict_parts_base(obj):
    return (syms(obj), [sym_predict(obj)], None)

sym_predict_parts_dispatcher = {}
sym_predict_parts = fallback(call_method_or_dispatch('sym_predict_parts', sym_predict_parts_dispatcher), sym_predict_parts_base)

sym_transform_dispatcher = {Earth: export_sympy_term_expressions}
sym_transform = call_method_or_dispatch('sym_transform', sym_transform_dispatcher)

def sym_transform_parts_base(obj, target=None):
    return (syms(obj), sym_transform(obj), target)

sym_transform_parts_dispatcher = {}
sym_transform_parts = fallback(call_method_or_dispatch('sym_transform_parts', sym_transform_parts_dispatcher), sym_transform_parts_base)

def assert_parts_are_composable(parts):
    inputs, expressions, target = parts
    assert set(inputs) >= set(chain(*map(lambda x:x.free_symbols, expressions)))
    if target is not None:
        target_inputs, _, _ = target
        assert len(target_inputs) == len(expressions) 
        assert_parts_are_composable(target)

def assemble_parts_into_expressions(parts):
    inputs, expressions, target = parts
    if target is not None:
        target_inputs, target_expressions, target_target = target
        assert len(target_inputs) == len(expressions)
        composed_expressions = [expr.subs(dict(zip(target_inputs, expressions))) for expr in target_expressions]
        return assemble_parts_into_expressions((inputs, composed_expressions, target_target))
    else:
        return inputs, expressions
    
def assemble_parts_into_assignment_pairs_and_outputs(parts):
    _, expressions, target = parts
    result = []
    if target is not None:
        target_inputs, _, _ = target
        assert len(target_inputs) == len(expressions)
        result.extend(zip(target_inputs, expressions))
        target_result, outputs = assemble_parts_into_assignment_pairs_and_outputs(target)
        result.extend(target_result)
        return result, outputs
    else:
        return result, expressions

    
sym_update_dispatcher = {}
sym_update = fallback(call_method_or_dispatch('sym_update', sym_update_dispatcher), sym_transform)

class STJavaScriptPrinter(JavascriptCodePrinter):
    def _print_Max(self, expr):
        return 'Math.max(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Min(self, expr):
        return 'Math.min(' + ','.join(self._print(i) for i in expr.args) + ')'
 
    def _print_NaNProtect(self, expr):
        return 'nanprotect(' + ','.join(self._print(i) for i in expr.args) + ')'
 
    def _print_Missing(self, expr):
        return 'missing(' + ','.join(self._print(a) for a in expr.args) + ')'
    
javascript_template_filename = os.path.join(resources, 'javascript_template.mako.js')
with open(javascript_template_filename) as infile:
    javascript_template = Template(infile.read())

def javascript_str(function_name, estimator, method=sym_predict, all_variables=False):
    expression = method(estimator)
    used_names = expression.free_symbols
    input_names = [sym.name for sym in syms(estimator) if sym in used_names or all_variables]
    return javascript_template.render(function_name=function_name, input_names=input_names,
                                      function_code=STJavaScriptPrinter().doprint(expression))

class STNumpyPrinter(NumPyPrinter):
    def _print_Max(self, expr):
        return 'maximum(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Min(self, expr):
        return 'minimum(' + ','.join(self._print(i) for i in expr.args) + ')'

    def _print_NaNProtect(self, expr):
        return 'where(isnan(' + ','.join(self._print(a) for a in expr.args) + '), 0, ' \
            + ','.join(self._print(a) for a in expr.args) + ')'

    def _print_Missing(self, expr):
        return 'isnan(' + ','.join(self._print(a) for a in expr.args) + ').astype(float)'
    
numpy_template_filename = os.path.join(resources, 'numpy_template.mako.py')
with open(numpy_template_filename) as infile:
    numpy_template = Template(infile.read())

def numpy_str(function_name, estimator, method=sym_predict, all_variables=False, pep8=False):
    expression = method(estimator)
    used_names = expression.free_symbols
    input_names = [sym.name for sym in syms(estimator) if sym in used_names or all_variables]
    function_code = STNumpyPrinter().doprint(expression)
    result = numpy_template.render(function_name=function_name, input_names=input_names,
                                      function_code=function_code)
    if pep8:
        result =  autopep8.fix_code(result, options={'aggressive': 1})
    return result

class STPythonPrinter(PythonPrinter):
    def _print_Float(self, expr):
        return str(expr)
    
    def _print_Not(self, expr):
        return 'negate(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Max(self, expr):
        return 'max(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_Min(self, expr):
        return 'min(' + ','.join(self._print(i) for i in expr.args) + ')'
    
    def _print_NaNProtect(self, expr):
        return 'nanprotect(' + ','.join(self._print(i) for i in expr.args) + ')'

    def _print_Missing(self, expr):
        return 'missing(' + ','.join(self._print(i) for i in expr.args) + ')'

python_template_filename = os.path.join(resources, 'python_template.mako.py')
with open(python_template_filename) as infile:
    python_template = Template(infile.read())

def python_str(function_name, estimator, method=sym_predict, all_variables=False):
    expression = method(estimator)
    used_names = expression.free_symbols
    input_names = [sym.name for sym in syms(estimator) if sym in used_names or all_variables]
    return autopep8.fix_code(python_template.render(function_name=function_name, input_names=input_names,
                                      function_code=STPythonPrinter().doprint(expression)), options={'aggressive': 1})


language_print_dispatcher = {
    'python': STPythonPrinter,
    'numpy': STNumpyPrinter,
    'javascript': STJavaScriptPrinter
    }

language_template_dispatcher = {
    'python': python_template,
    'numpy': numpy_template,
    'javascript': javascript_template 
    }

language_assignment_statement_dispatcher = defaultdict(lambda: lambda symbol, expression: symbol + ' = ' + expression)

language_return_statement_dispatcher = defaultdict(lambda: lambda expressions: 'return ' + ', '.join(expressions))

def trim_code_precursors(assignments, outputs, inputs, all_variables):
    reverse_new_assignments = []
    new_inputs = []
    used = set(reduce(add, map(lambda x: x.free_symbols, outputs)))
    for variable, expr in reversed(assignments):
        if variable in used:
            used.update(expr.free_symbols)
            reverse_new_assignments.append((variable, expr))
    if not all_variables:
        for variable in inputs:
            if variable in used:
                new_inputs.append(variable)
    else:
        new_inputs.extend(inputs)
    return reversed(reverse_new_assignments), new_inputs
            
            

def assignment_pairs_and_outputs_to_code(pairs_and_outputs, language, function_name, inputs, all_variables):
    assignments, outputs = pairs_and_outputs
    assignment_statements = ''
    assignments, inputs_ = trim_code_precursors(assignments, outputs, inputs, all_variables)
    
    printer = language_print_dispatcher[language]
    assigner = language_assignment_statement_dispatcher[language]
    returner = language_return_statement_dispatcher[language]
    template = language_template_dispatcher[language]
    for symbol, expression in assignments:
        
        assignment_statements += assigner(symbol.name, printer().doprint(expression)) + '\n'
    
    return_statement = returner(map(printer().doprint, outputs))
    return template.render(function_name=function_name, input_names=map(lambda x: x.name, inputs_), 
                           assignment_code=assignment_statements, return_code=return_statement)
    
def parts_to_code(parts, language, function_name, all_variables):
    pairs_and_outputs = assemble_parts_into_assignment_pairs_and_outputs(parts)
    inputs = [symbol for symbol in parts[0]]
    return assignment_pairs_and_outputs_to_code(pairs_and_outputs, language, function_name, inputs, all_variables)

model_to_code_method_dispatch = {'predict': sym_predict_parts,
                                 'transform': sym_transform_parts}

def model_to_code(model, language, method, function_name, all_variables=False):
    parts = model_to_code_method_dispatch[method](model)
    result = parts_to_code(parts, language, function_name, all_variables)
    return result
    
    
