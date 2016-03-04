'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin, is_classifier,\
    is_regressor, TransformerMixin
import numpy as np
from sklearn.cross_validation import check_cv, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import safe_mask
from six import with_metaclass
from operator import attrgetter
from functools import update_wrapper
from types import MethodType

class SklearnTool(object):
    pass

def name_estimator(estimator):
    if hasattr(estimator, 'name'):
        return estimator.name
    else:
        return estimator.__class__.__name__

def combine_named_estimators(names):
    used_set = set()
    result = []
    for name in names:
        new_name = name
        i = 2
        while new_name in used_set:
            new_name = name + '_' + str(i)
            i += 1
            if i > 1e5:
                raise ValueError('Unable to name estimator %s in pipeline' % str(name))
        used_set.add(new_name)
        result.append(new_name)
    return result

class STEstimator(BaseEstimator, SklearnTool):
    def mask(self, mask):
        return mask_estimator(self, mask)

    def __rshift__(self, other):
        '''
        self >> other
        '''
        return as_pipeline(self) >> other
        
    def __rrshift__(self, other):
        '''
        other >> self
        '''
        return as_pipeline(other) >> self
        
    def __lshift__(self, other):
        '''
        self << other
        '''
        return as_pipeline(other) >> self
        
    def __rlshift__(self, other):
        '''
        other << self
        '''
        return as_pipeline(self) >> other
    
    def __and__(self, other):
        '''
        self & other
        '''
        mask_estimator(self, slice(None)) & other
    
    def __rand__(self, other):
        '''
        other & self
        '''
        return self & other
        
def as_pipeline(estimator):
    try:
        return estimator.as_pipeline()
    except AttributeError:
        return STPipeline([(name_estimator(estimator), estimator)])

class STSimpleEstimator(STEstimator):
    def as_pipeline(self):
        return STPipeline([(name_estimator(self), self)])
        
class STPipeline(STEstimator, Pipeline):
    def as_pipeline(self):
        return self
    
    def __rshift__(self, other):
        '''
        self >> other
        '''
        other = as_pipeline(other)
        steps = self.steps + other.steps
        names = [step[0] for step in steps]
        estimators = [step[1] for step in steps]
        return STPipeline(zip(combine_named_estimators(names), estimators))

#         
class _BasicDelegateDescriptor(object):
    def __init__(self, fn, delegate_name):
        self.fn = fn
        self.delegate_name = delegate_name
        self.method_name = fn.__name__
        # update the docstring of the descriptor
        update_wrapper(self, fn)
        
    def __get__(self, obj, type=None):  # @ReservedAssignment
        if self.delegate_name is None:
            try:
                delegate_name = obj._delegates[self.method_name]
            except KeyError:
                try:
                    delegate_name = obj.__class__._class_delegates[self.method_name]
                except KeyError:
                    raise AttributeError()
        else:
            delegate_name = self.delegate_name
        clone_name = delegate_name + '_'
        
        # If the clone doesn't exist, it needs to be created by this call
        if not hasattr(obj, clone_name):
            delegate = getattr(obj, delegate_name)
            method = getattr(delegate, self.method_name)
            def out(*args, **kwargs):
                setattr(obj, clone_name, method(*args, **kwargs))
                return obj
        else:
            delegate = getattr(obj, clone_name)
            method = getattr(delegate, self.method_name)
            out = lambda *args, **kwargs: method(*args, **kwargs)
        update_wrapper(out, self.fn)
        return out

def delegate_by_name(delegate_name=None):
    return lambda fn: _BasicDelegateDescriptor(fn, delegate_name)

def delegate(fn):
    return _BasicDelegateDescriptor(fn, None)

# def delegate_init(fn):
#     def init(self, *args, **kwargs):
#         self._delegates = {}
#         fn(self, *args, **kwargs)
#     update_wrapper(init, fn)
#     return init

class DelegatingMetaClass(type):
    '''
    Every subclass gets its own _delegates dictionary.  If the dictionary were
    just a normal class attribute on BaseDelegatingEstimator, it would be shared
    among all subclasses.
    '''
    def __init__(cls, name, bases, dict):  # @ReservedAssignment
        super(DelegatingMetaClass, cls).__init__(name, bases, dict)
        cls._class_delegates = {}
#         cls.__init__ = delegate_init(cls.__init__)
        
#     def __call__(cls, *args, **kwargs):  # @NoSelf
#         obj = super(DelegatingMetaClass, cls).__call__(*args, **kwargs)
#         obj._delegates = {}

standard_methods = ['fit', 'predict', 'score', 'predict_proba', 'decision_function', 
                         'predict_log_proba', 'transform']
non_fit_methods = ['predict', 'score', 'predict_proba', 'decision_function', 
                         'predict_log_proba', 'transform']
predict_methods = ['predict', 'predict_proba', 'decision_function', 
                         'predict_log_proba']

class BaseDelegatingEstimator(with_metaclass(DelegatingMetaClass, STSimpleEstimator, MetaEstimatorMixin)):
    def _create_delegates(self, name, method_names):
        if not hasattr(self, '_delegates'):
            self.delegates = {}
        delegate_ = getattr(self, name)
        methods = [method for method in method_names if callable(getattr(delegate_, method, None))]
        for method in methods:
            self._delegates[method] = name
#             def fn(obj):
#                 pass
#             fn.__name__ = method
#             setattr(self, method, delegate()(MethodType(fn, self, self.__class__)))

    @delegate
    def fit(self):
        pass
     
    @delegate
    def predict(self):
        pass
     
    @delegate
    def score(self):
        pass
     
    @delegate
    def predict_proba(self):
        pass
     
    @delegate
    def decision_function(self):
        pass
     
    @delegate
    def predict_log_proba(self):
        pass
     
    @delegate
    def transform(self):
        pass

class DelegatingEstimator(BaseDelegatingEstimator):
    _delegates = {'fit': 'estimator', 'predict': 'estimator', 'score': 'estimator', 
                  'predict_proba': 'estimator', 'decision_function': 'estimator', 
                  'predict_log_proba': 'estimator'}
    def __init__(self, estimator):
        self.estimator = estimator

class AlreadyFittedEstimator(DelegatingEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_ = self.estimator
        self._create_delegates('estimator', non_fit_methods)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        return self
    
def mask_estimator(estimator, mask):
    try:
        return estimator.mask(mask)
    except AttributeError:
        return MultipleResponseEstimator([(name_estimator(estimator), mask, estimator)])

def is_masked(estimator):
    return isinstance(estimator, MultipleResponseEstimator)

def compatible_masks(masks):
    # (min, max)
    intersection = (0, float('inf'))
    for mask in masks:
        current_mask = mask
        if isinstance(current_mask, np.ndarray):
            if current_mask.dtype.kind == 'b':
                current = (current_mask.shape[0], current_mask.shape[0])
            else:
                current = (np.max(current_mask), float('inf'))
        elif isinstance(current_mask, slice):
            current = (max(current_mask.start, current_mask.stop), float('inf'))
        intersection = (max(current[0], intersection[0]), min(current[1], intersection[1]))
        
    if intersection[0] > intersection[1]:
        return False
    else:
        return True
    
def convert_mask(mask):
    if not (isinstance(mask, np.ndarray) or isinstance(mask, slice)) \
            and hasattr(mask, '__iter__'):
        return np.array(mask)
    else:
        return mask

class MultipleResponseEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimators):
        names = [est[0] for est in estimators]
        masks = [convert_mask(est[1]) for est in estimators]
        if not compatible_masks(masks):
            raise ValueError('Masks do not have compatible sizes')
        estimators_ = [est[2] for est in estimators]
        self.estimators = zip(names, masks, estimators_)
    
    def __and__(self, other):
        other = mask_estimator(other, slice(None))
        all_estimators = self.estimators + other.estimators
        names = [est[0] for est in all_estimators]
        masks = [est[1] for est in all_estimators]
        estimators = [est[2] for est in all_estimators]
        names = combine_named_estimators(names)
        if not compatible_masks(masks):
            raise ValueError('Masks do not have compatible sizes')
        return MultipleResponseEstimator(zip(names, masks, estimators))
        
    def mask(self, mask):
        assert mask == slice(None)
        return self
    
    def _masks(self):
        return [mask for _, mask, _ in self.estimators]
    
    @property
    def _estimator_type(self):
        if all([is_classifier(estimator) for estimator in self.estimators.values()]):
            return 'classifier'
        elif all([is_regressor(estimator) for estimator in self.estimators.values()]):
            return 'regressor'
        else:
            return 'mixed'
    
    def fit(self, X, y, fit_args=None, *args, **kwargs):
        if fit_args is None:
            fit_args = {}
        self.estimators_ = []
        for name, columns, model in self.estimators:
            dargs = kwargs.copy()
            dargs.update(fit_args.get(name, {}))
            
            # Select the appropriate columns
            y_ = y[:, columns]
            if y_.shape[1] == 1:
                y_ = y_[:, 0]
            
            # Fit the estimator
            self.estimators_.append((name, columns, clone(model).fit(X, y_, *args, **dargs)))
        self.estimators_dict_ = {name: (columns, model) for name, columns, model in self.estimators_}
        
        # Do a prediction on a single row of data for each estimator in order to 
        # determine the number of predicted columns for each one
        X_ = X[0:1, :]
        self.prediction_columns_ = []
        for name, columns, model in self.estimators_:
            prediction = model.predict(X_)
            if len(prediction.shape) > 1:
                n_columns = prediction.shape[1]
            else:
                n_columns = 1
            self.prediction_columns_ += [name] * n_columns
        
        return self
    
    def predict(self, X, predict_args=None, *args, **kwargs):
        if predict_args is None:
            predict_args = {}
        predictions = []
        for name, columns, model in self.estimators_:  # @UnusedVariable
            dargs = kwargs.copy()
            dargs.update(predict_args.get(name, {}))
            prediction = model.predict(X, *args, **dargs)
            predictions.append(prediction if len(prediction.shape) == 2 else prediction[:, None])
        return np.concatenate(predictions, axis=1)
    
    def predict_proba(self, X, predict_args=None, *args, **kwargs):
        if predict_args is None:
            predict_args = {}
        predictions = []
        for name, columns, model in self.estimators_:  # @UnusedVariable
            dargs = kwargs.copy()
            dargs.update(predict_args.get(name, {}))
            prediction = model.predict_proba(X, *args, **dargs)
            predictions.append(prediction if len(prediction.shape) == 2 else prediction[:, None])
        return np.concatenate(predictions, axis=1)



