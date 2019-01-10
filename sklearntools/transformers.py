from .sklearntools import STSimpleEstimator, safe_assign_subset, safe_column_names, clean_column_name, \
    safe_column_select, safe_assign_column
from abc import ABCMeta, abstractmethod
from sympy.core.symbol import Symbol
from toolz.dicttoolz import keymap, itemmap, valfilter
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.numbers import RealNumber, One, Zero
import numpy as np
from sympy import log as SymLog, Min as SymMin, Max as SymMax
from sympy.core.relational import Eq
from operator import __or__, methodcaller, __ge__
from toolz.functoolz import curry, compose
from six import string_types
from sklearn2code.sym.base import sym_predict as s2c_sym_predict, \
    sym_transform as s2c_sym_transform, syms as s2c_syms, sym_predict_proba as s2c_sym_predict_proba
from sklearn2code.dispatching import call_method_or_dispatch
from sklearn2code.sym.expression import RealNumber as S2CRealNumber,\
    RealVariable, RealPiecewise, true, Log as S2CLog, MinReal, LessReal,\
    LessEqualReal, GreaterReal, GreaterEqualReal, MaxReal, nan,\
    IsNan
from sklearn2code.sym.function import Function
from _collections import defaultdict
from sklearn.base import clone
from six.moves import reduce
from sklearntools.adapters import predict

sym_col_trans = call_method_or_dispatch('sym_col_trans', docstring='')

class ColumnTransformation(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def transform(self, X):
        pass
    
    @abstractmethod
    def sym_transform(self):
        pass
    
    @abstractmethod
    def inputs(self):
        pass
    
    def __call__(self, other):
        return Composition(self, other)
    
    def __add__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return Sum(self, other)
    
    def __radd__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other + self
    
    def __sub__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return Sum(self, Negative(other))
    
    def __rsub__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other - self
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    def __div__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return Quotient(self, other)
    
    def __rdiv__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other / self
        
    def __mul__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return Product(self, other)
    
    def __rmul__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other * self
    
    def __pow__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return Power(self, other)
    
    def __rpow__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other ** self
    
    def __lt__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return LT(self, other)
    
    def __rlt__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other < self
    
    def __le__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return LE(self, other)
    
    def __rle__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other <= self
    
    def __gt__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return GT(self, other)
    
    def __rgt__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other > self
    
    def __ge__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return GE(self, other)
    
    def __rge__(self, other):
        if not isinstance(other, ColumnTransformation):
            other = Constant(other)
        return other >= self
    
class Constant(ColumnTransformation):
    def __init__(self, value):
        self.value = value
    
    def inputs(self):
        return set()
    
    def transform(self, X):
        return np.ones(shape=X.shape[0]) * self.value
    
#     def sym_transform(self, xlabels):
#         return RealNumber(self.value)

@sym_col_trans.register(Constant)
def sym_constant(estimator):
    return S2CRealNumber(estimator.value)

class Identity(ColumnTransformation):
    def __init__(self, column):
        self.column = column
    
    def inputs(self):
        return set([self.column])
    
    def transform(self, X):
        return safe_column_select(X, self.column)
    
#     def sym_transform(self, xlabels):
#         return Symbol(clean_column_name(xlabels, self.column))

@sym_col_trans.register(Identity)
def sym_identity(estimator):
    return RealVariable(estimator.column)

class OneArgumentColumnTransformation(ColumnTransformation):
    def __init__(self, arg):
        self.arg = arg
    
    def inputs(self):
        return self.arg.inputs()
    
class Nonzero(OneArgumentColumnTransformation):
    def transform(self, X):
        return self.arg.transform(X) != 0
    
#     def sym_transform(self, xlabels):
#         arg = self.arg.sym_transform(xlabels)
#         return Piecewise((Zero(), Eq(arg, Zero())), (One(), True))

@sym_col_trans.register(Nonzero)
def sym_nonzero(estimator):
    return RealPiecewise((S2CRealNumber(0), (sym_col_trans(estimator.arg).e == S2CRealNumber(0))), (S2CRealNumber(1), true))

class Log(OneArgumentColumnTransformation):
    def transform(self, X):
        return np.log(self.arg.transform(X))
    
#     def sym_transform(self, xlabels):
#         arg = self.arg.sym_transform(xlabels)
#         return SymLog(arg)

@sym_col_trans.register(Log)
def sym_log(estimator):
    return S2CLog(sym_col_trans(estimator.arg))


class Negative(OneArgumentColumnTransformation):
    def transform(self, X):
        return -self.arg.transform(X)
    
#     def sym_transform(self, xlabels):
#         arg = self.arg.sym_transform(xlabels)
#         return -arg

@sym_col_trans.register(Negative)
def sym_negative(estimator):
    return -(sym_col_trans(estimator.arg))

class TwoArgumentColumnTransformation(ColumnTransformation):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def inputs(self):
        return self.left.inputs() | self.right.inputs()

class Min(TwoArgumentColumnTransformation):
    def transform(self, X):
        return np.amin(self.left.transform(X), self.right.transform(X))
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return SymMin(left, right)

@sym_col_trans.register(Min)
def sym_min(estimator):
    return MinReal(sym_col_trans(estimator.left), sym_col_trans(estimator.right))

class LT(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) < self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return Piecewise((One(), left<right), (Zero(), True))

@sym_col_trans.register(LT)
def sym_lt(estimator):
    return LessReal(sym_col_trans(estimator.left), sym_col_trans(estimator.right))

class LE(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) <= self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return Piecewise((One(), left<=right), (Zero(), True))

@sym_col_trans.register(LE)
def sym_le(estimator):
    return LessEqualReal(sym_col_trans(estimator.left), sym_col_trans(estimator.right))

class GT(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) > self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return Piecewise((One(), left>right), (Zero(), True))

@sym_col_trans.register(GT)
def sym_gt(estimator):
    return GreaterReal(sym_col_trans(estimator.left), sym_col_trans(estimator.right))

class GE(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) >= self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return Piecewise((One(), left>=right), (Zero(), True))

@sym_col_trans.register(GE)
def sym_ge(estimator):
    return GreaterEqualReal(sym_col_trans(estimator.left), sym_col_trans(estimator.right))

class Max(TwoArgumentColumnTransformation):
    def transform(self, X):
        return np.amax(self.left.transform(X), self.right.transform(X))
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return SymMax(left, right)

@sym_col_trans.register(Max)
def sym_max(estimator):
    return MaxReal(sym_col_trans(estimator.left), sym_col_trans(estimator.right))

class Power(TwoArgumentColumnTransformation):
    def transform(self, X):
        return np.power(self.left.transform(X), self.right.transform(X))
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return left ** right#Piecewise((left**right, (left > 0)| (left < 0) | (right > 0)), (NAN(1), True))

@sym_col_trans.register(Power)
def sym_power(estimator):
    return sym_col_trans(estimator.left) ** sym_col_trans(estimator.right)

class Sum(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) + self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return left + right

@sym_col_trans.register(Sum)
def sym_sum(estimator):
    return sym_col_trans(estimator.left) + sym_col_trans(estimator.right)

class Product(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) * self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return left * right

@sym_col_trans.register(Product)
def sym_product(estimator):
    return sym_col_trans(estimator.left) * sym_col_trans(estimator.right)

class Quotient(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) / self.right.transform(X)
    
#     def sym_transform(self, xlabels):
#         left = self.left.sym_transform(xlabels)
#         right = self.right.sym_transform(xlabels)
#         return left / right

@sym_col_trans.register(Quotient)
def sym_quotient(estimator):
    return sym_col_trans(estimator.left) / sym_col_trans(estimator.right)

class Composition(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(self.right.transform(X))
    
#     def sym_transform(self, xlabels):
#         return self.left.sym_transform(xlabels).subs({self.right.name: self.right.sym_transform(xlabels)})

@sym_col_trans.register(Composition)
def sym_composition(estimator):
    return sym_col_trans(estimator.left).subs({estimator.right.name: sym_col_trans(estimator.right)})

class Censor(TwoArgumentColumnTransformation):
    def transform(self, X):
        result = self.left.transform(X).copy()
        safe_assign_subset(result, self.right.transform(X) != 0, np.nan)
        return result
    
@sym_col_trans.register(Censor)
def sym_censor(estimator):
    left = sym_col_trans(estimator.left)
    right = sym_col_trans(estimator.right)
    return RealPiecewise((nan, right.e == S2CRealNumber(1)), (left, true))

class Uncensor(TwoArgumentColumnTransformation):
    def transform(self, X):
        result = self.left.transform(X).copy()
        safe_assign_subset(result, np.isnan(result), self.right.transform(X))
        return result
    
@sym_col_trans.register(Uncensor)
def sym_uncensor(estimator):
    left = sym_col_trans(estimator.left)
    right = sym_col_trans(estimator.right)
    return RealPiecewise((right, IsNan(left)), (left, true))

class VariableTransformer(STSimpleEstimator):
    def __init__(self, transformations, strict=False, exclusive=False):
        '''
        strict : (bool) If True, fail on missing inputs.  If False, just
        skip them.
        
        exclusive : (bool) If True, output only the results of transformations and
        not the original input data.
        '''
        self.transformations = transformations
        self.strict = strict
        self.exclusive = exclusive
    
    def inputs(self):
        return reduce(__or__, map(methodcaller('inputs'), self.transformations.values()))
    
    def fit(self, X, y=None, exposure=None, xlabels=None):
        if xlabels is not None:
            self.xlabels_ = xlabels
        else:
            self.xlabels_ = safe_column_names(X)
        self.clean_transformations_ = keymap(clean_column_name(self.xlabels_), self.transformations)
        if not self.strict:
            input_variables = set(self.xlabels_)
            self.clean_transformations_ = valfilter(compose(curry(__ge__)(input_variables), methodcaller('inputs')), self.clean_transformations_)
        return self
    
    def transform(self, X, offset=None, exposure=None):
        if self.exclusive:
            result = type(X)()
        else:
            result = X.copy()
        for k, v in self.clean_transformations_.items():
            safe_assign_column(result, k, v.transform(X))
        return result
    
#     def syms(self):
#         return [Symbol(label) for label in self.xlabels_]
#     
#     def sym_transform(self):
#         input_syms = self.syms() 
#         syms = input_syms + list(filter(lambda x: x not in input_syms, map(Symbol, self.clean_transformations_.keys())))
#         result = []
#         for sym in syms:
#             name = sym.name
#             if name in self.clean_transformations_:
#                 result.append(self.clean_transformations_[name].sym_transform(self.xlabels_))
#             else:
#                 result.append(sym)
#         return result

class TransformingEstimator(STSimpleEstimator):
    def __init__(self, estimator, x_transformer=None, y_transformer=None, 
                 exposure_transformer=None, weight_transformer=None):
        self.estimator = estimator
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer
        self.exposure_transformer = exposure_transformer
        self.weight_transformer = weight_transformer
    
    def inputs(self):
        return self.x_transformer.inputs()
    
    def outputs(self):
        return self.y_transformer.inputs()
    
    def _internal_transform(self, X, include_response):
        args = {}
        if self.x_transformer is None:
            args['X'] = X
        else:
            args['X'] = self.x_transformer_.transform(X)
        if self.exposure_transformer is not None:
            args['exposure'] = self.exposure_transformer_.transform(X)
        if include_response:
            if self.y_transformer is not None:
                args['y'] = self.y_transformer_.transform(X)
            if self.weight_transformer is not None:
                args['sample_weight'] = self.weight_transformer_.transform(X)
        return args
    
    def fit(self, X):
        if self.x_transformer is not None:
            self.x_transformer_ = clone(self.x_transformer).fit(X)
        if self.y_transformer is not None:
            self.y_transformer_ = clone(self.y_transformer).fit(X)
        if self.exposure_transformer is not None:
            self.exposure_transformer_ = clone(self.exposure_transformer).fit(X)
        if self.weight_transformer is not None:
            self.weight_transformer_ = clone(self.weight_transformer).fit(X)
        args = self._internal_transform(X, True)
        try:
            self.estimator_ = clone(self.estimator).fit(**args)
        except:
            self.estimator_ = clone(self.estimator).fit(**args)
        return self
    
    def transform(self, X):
        args = self._internal_transform(X, False)
        return self.estimator_.transform(**args)
    
    def predict(self, X):
        args = self._internal_transform(X, False)
        return predict(self.estimator_, **args)
#         try:
#             return self.estimator_.predict(**args)
#         except:
#             return self.estimator_.predict(**args)
    
    def predict_proba(self, X):
        args = self._internal_transform(X, False)
        return self.estimator_.predict_proba(**args)
    

@s2c_syms.register(TransformingEstimator)
def syms_transforming_estimator(estimator):
    if estimator.x_transformer is None:
        return s2c_syms(estimator.estimator_)
    else:
        return s2c_syms(estimator.x_transformer_)

@s2c_sym_transform.register(TransformingEstimator)
def sym_transform_transforming_estimator(estimator):
    if estimator.x_transformer is None:
        return s2c_sym_transform(estimator.estimator_)
    else:
        return s2c_sym_transform(estimator.estimator_).compose(s2c_sym_transform(estimator.x_transformer_))

@s2c_sym_predict.register(TransformingEstimator)
def sym_predict_tranforming_estimator(estimator):
    if estimator.x_transformer is None:
        return s2c_sym_predict(estimator.estimator_)
    else:
        return s2c_sym_predict(estimator.estimator_).compose(s2c_sym_transform(estimator.x_transformer_))

@s2c_sym_predict_proba.register(TransformingEstimator)
def sym_predict_proba_tranforming_estimator(estimator):
    if estimator.x_transformer is None:
        return s2c_sym_predict_proba(estimator.estimator_)
    else:
        return s2c_sym_predict_proba(estimator.estimator_).compose(s2c_sym_transform(estimator.x_transformer_))

@s2c_syms.register(VariableTransformer)
def syms_variable_transformer(estimator):
    return tuple(map(RealVariable, estimator.xlabels_))

@s2c_sym_transform.register(VariableTransformer)
def sym_transform_variable_transformer(estimator):
    inputs = s2c_syms(estimator)
    label_to_idxs = defaultdict(list)
    for i, label in enumerate(estimator.xlabels_):
        label_to_idxs[label].append(i)
    
    label_to_expr = dict(map(lambda label: (label, RealVariable(label)), estimator.xlabels_))
    label_to_expr.update(estimator.clean_transformations_)
    
    outputs = list(inputs)
    for label, expr in estimator.clean_transformations_.items():
        if label in label_to_idxs:
            idxs = label_to_idxs[label]
            for idx in idxs:
                outputs[idx] = sym_col_trans(expr)
        else:
            outputs.append(sym_col_trans(expr))
    
    outputs = tuple(outputs)
    
#     outputs_dict = OrderedDict(map(lambda label: (label, RealVariable(label)), estimator.xlabels_))
#     print(len(outputs_dict))
#     print(len(estimator.xlabels_))
#     print(len(estimator.clean_transformations_))
#     for k, v in estimator.clean_transformations_.items():
#         outputs_dict[k] = sym_col_trans(v)
#     print(len(outputs_dict))
#     outputs = tuple(outputs_dict.values())#tuple(valmap(sym_col_trans, estimator.clean_transformations_).values())
#     print(len(outputs))
    return Function(inputs, tuple(), outputs)

def NanMap(nan_map, strict=False):
    return VariableTransformer(itemmap(lambda tup: 
                                       (tup[0], Uncensor(Identity(tup[0]), Identity(tup[1]) if isinstance(tup[1], string_types) 
                                                       else Constant(tup[1]))), nan_map), strict=strict)


