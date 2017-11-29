from .sklearntools import STSimpleEstimator, safe_assign_subset, safe_column_names, clean_column_name, \
    safe_column_select, safe_assign_column
from abc import ABCMeta, abstractmethod
from sympy.core.symbol import Symbol
from toolz.dicttoolz import keymap, valmap, itemmap, valfilter
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.numbers import RealNumber, One, Zero
import numpy as np
from sympy import log as SymLog, Min as SymMin, Max as SymMax
from .sym.base import NAN, Missing
from sympy.core.relational import Eq
from operator import __or__, methodcaller, __ge__
from toolz.functoolz import curry, compose

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
    
    def sym_transform(self, xlabels):
        return RealNumber(self.value)

class Identity(ColumnTransformation):
    def __init__(self, column):
        self.column = column
    
    def inputs(self):
        return set([self.column])
    
    def transform(self, X):
        return safe_column_select(X, self.column)
    
    def sym_transform(self, xlabels):
        return Symbol(clean_column_name(xlabels, self.column))

class OneArgumentColumnTransformation(ColumnTransformation):
    def __init__(self, arg):
        self.arg = arg
    
    def inputs(self):
        return self.arg.inputs()
    
class Nonzero(OneArgumentColumnTransformation):
    def transform(self, X):
        return self.arg.transform(X) != 0
    
    def sym_transform(self, xlabels):
        arg = self.arg.sym_transform(xlabels)
        return Piecewise((Zero(), Eq(arg, Zero())), (One(), True))

class Log(OneArgumentColumnTransformation):
    def transform(self, X):
        return np.log(self.arg.transform(X))
    
    def sym_transform(self, xlabels):
        arg = self.arg.sym_transform(xlabels)
        return SymLog(arg)
        
class Negative(OneArgumentColumnTransformation):
    def transform(self, X):
        return -self.arg.transform(X)
    
    def sym_transform(self, xlabels):
        arg = self.arg.sym_transform(xlabels)
        return -arg

class TwoArgumentColumnTransformation(ColumnTransformation):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def inputs(self):
        return self.left.inputs() | self.right.inputs()

class Min(TwoArgumentColumnTransformation):
    def transform(self, X):
        return np.amin(self.left.transform(X), self.right.transform(X))
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return SymMin(left, right)

class LT(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) < self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return Piecewise((One(), left<right), (Zero(), True))

class LE(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) <= self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return Piecewise((One(), left<=right), (Zero(), True))
    
class GT(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) > self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return Piecewise((One(), left>right), (Zero(), True))

class GE(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) >= self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return Piecewise((One(), left>=right), (Zero(), True))
    
class Max(TwoArgumentColumnTransformation):
    def transform(self, X):
        return np.amax(self.left.transform(X), self.right.transform(X))
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return SymMax(left, right)

class Power(TwoArgumentColumnTransformation):
    def transform(self, X):
        return np.power(self.left.transform(X), self.right.transform(X))
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return left ** right#Piecewise((left**right, (left > 0)| (left < 0) | (right > 0)), (NAN(1), True))

class Sum(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) + self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return left + right

class Product(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) * self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return left * right

class Quotient(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(X) / self.right.transform(X)
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return left / right

class Composition(TwoArgumentColumnTransformation):
    def transform(self, X):
        return self.left.transform(self.right.transform(X))
    
    def sym_transform(self, xlabels):
        return self.left.sym_transform(xlabels).subs({self.right.name: self.right.sym_transform(xlabels)})

class Censor(TwoArgumentColumnTransformation):
    def transform(self, X):
        result = self.left.transform(X).copy()
        safe_assign_subset(result, self.right.transform(X) != 0, np.nan)
        return result
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return Piecewise((NAN(1), Eq(right, One())), (left, True))

class Uncensor(TwoArgumentColumnTransformation):
    def transform(self, X):
        result = self.left.transform(X).copy()
        safe_assign_subset(result, np.isnan(result), self.right.transform(X))
        return result
    
    def sym_transform(self, xlabels):
        left = self.left.sym_transform(xlabels)
        right = self.right.sym_transform(xlabels)
        return Piecewise((right, Eq(Missing(left), One())), (left, True))

class VariableTransformer(STSimpleEstimator):
    def __init__(self, transformations, strict=False):
        '''
        strict : (bool) If True, fail on missing inputs.  If False, just
        skip them.
        '''
        self.transformations = transformations
        self.strict = strict
    
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
        result = X.copy()
        for k, v in self.clean_transformations_.items():
            safe_assign_column(result, k, v.transform(X))
        return result
    
    def syms(self):
        return [Symbol(label) for label in self.xlabels_]
    
    def sym_transform(self):
        input_syms = self.syms() 
        syms = input_syms + filter(lambda x: x not in input_syms, map(Symbol, self.clean_transformations_.keys()))
        result = []
        for sym in syms:
            name = sym.name
            if name in self.clean_transformations_:
                result.append(self.clean_transformations_[name].sym_transform(self.xlabels_))
            else:
                result.append(sym)
        return result

def NanMap(nan_map, strict=False):
    return VariableTransformer(itemmap(lambda (name, val): 
                                       (name, Uncensor(Identity(name), Identity(val) if isinstance(val, basestring) 
                                                       else Constant(val))), nan_map), strict=strict)


