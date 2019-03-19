from .sklearntools import STSimpleEstimator, growd
from operator import methodcaller, __add__
from sklearn.base import clone
from toolz.functoolz import compose, flip
import pandas
import numpy as np
from collections import OrderedDict
from six.moves import reduce

# def best_type(types, arg, *args):
#     while not isinstance(arg, types[-1]):
#         types.pop()
#         if not types:
#             raise TypeError('No best type found.')
#     if not args:
#         return types[-1]
#     return best_type(types, *args)
        
def safe_concat(*args):
    types = set(map(type, args))
    if not types <= {np.ndarray, pandas.DataFrame}:
        raise TypeError('Inputs must be either pandas DataFrames or numpy ndarrays.')
    if np.ndarray in types:
        return_type = np.ndarray
    else:
        return_type = pandas.DataFrame
#     try:
#         return_type = best_type(
#                                 [np.ndarray, pandas.DataFrame],
#                                 *args
#                                 )
#     except TypeError:
#         raise TypeError('Inputs must be either pandas DataFrames or numpy ndarrays.')
    if return_type is np.ndarray:
        return np.concatenate(tuple(map(growd(2), args)), axis=1)
    else:
        columns = reduce(__add__, map(compose(list, flip(getattr)('columns')), args))
        result = pandas.concat(args, axis=1, ignore_index=True)
        result.columns = columns
        return result
        
    
class ConcatenatingEstimator(STSimpleEstimator):
    def __init__(self, estimators):
        self.estimators = tuple(estimators)
        self.ordered_estimators = OrderedDict(self.estimators.items() if hasattr(self.estimators, 'items') else self.estimators)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        self.estimators_ = OrderedDict(
                                       zip(
                                           self.ordered_estimators.keys(), 
                                           map(
                                               compose(
                                                       methodcaller('fit', **fit_args), 
                                                       clone
                                                       ),
                                               self.ordered_estimators.values()
                                               )
                                           )
                                       )
        return self
    
    def predict(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        return safe_concat(*map(methodcaller('predict', **args), self.estimators_.values()))
    
    def transform(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        return safe_concat(*map(methodcaller('transform', **args), self.estimators_.values()))
    
    