'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
import numpy as np
from sklearn.utils.metaestimators import if_delegate_has_method
from six import with_metaclass
from functools import update_wrapper
from inspect import getargspec
from sym import syms, sym_update, sym_predict

def safe_call(fn, args):
    if hasattr(fn, '_spec'):
        spec = fn._spec
    else:   
        spec = getargspec(fn)
    if spec.keywords is not None:
        return fn(**args)
    else:
        safe_args = {arg: args[arg] for arg in spec.args[1:] if arg in args}
        try:
            return fn(**safe_args)
        except:
            1+1
            return fn(**safe_args)
        
def safer_call(fn, *args, **kwargs):
    kwargs = kwargs.copy()
    try:
        spec = getargspec(fn)
    except TypeError:
        spec = getargspec(fn.__call__)
    spec_args = list(spec.args)
    if spec_args[0] == 'self':
        spec_args = spec_args[1:]
    if spec.varargs is None:
        for i, arg in enumerate(args):
            name = spec_args[i]
            kwargs[name] = arg
            args = []
    if spec.keywords is None:
        for name in list(kwargs.keys()):
            if name not in spec_args:
                del kwargs[name]
    return fn(*args, **kwargs)

def _subset(data, idx):
    if len(data.shape) == 1:
        return data[idx]
    else:
        if hasattr(data, 'loc'):
            return data.loc[idx, :]
        else:
            return data[idx, :]

def _subset_data(data, idx):
    result = {}
    for k in data.keys():
        result[k] = _subset(data[k], idx)
    return result

def _fit_and_score(estimator, data, scorer, train, test):
    train_data = _subset_data(data, train)
    estimator_ = clone(estimator).fit(**train_data)
    test_data = _subset_data(data, test)
    score = safer_call(scorer, estimator_, **test_data)
    return (score, np.sum(test))

class SklearnTool(object):
    pass
# 
# def name_estimator(estimator):
#     if hasattr(estimator, 'name'):
#         return estimator.name
#     else:
#         return estimator.__class__.__name__
# 
# def combine_named_estimators(names):
#     used_set = set()
#     result = []
#     for name in names:
#         new_name = name
#         i = 2
#         while new_name in used_set:
#             new_name = name + '_' + str(i)
#             i += 1
#             if i > 1e5:
#                 raise ValueError('Unable to name estimator %s in pipeline' % str(name))
#         used_set.add(new_name)
#         result.append(new_name)
#     return result

class STEstimator(BaseEstimator, SklearnTool):
    def _process_args(self, **kwargs):
        result = {}
        for k, v in kwargs.items():
            if v is not None:
                result[k] = v
        for k in result.keys():
            v = result[k]
            if isinstance(v, np.ndarray):
                if len(v.shape) == 1:
                    result[k] = v[:, None]
        return result
    
#     def mask(self, mask):
#         return mask_estimator(self, mask)

    def __and__(self, other):
        '''
        self & other
        '''
        return MultiEstimator([self]) & other
    
    def __rand__(self, other):
        '''
        other & self
        '''
        return MultiEstimator([other]) & self


class StagedEstimator(STEstimator, MetaEstimatorMixin):
    def __init__(self, stages):
        self.stages = stages
        self.intermediate_stages = self.stages[:-1]
        self.final_stage = self.stages[-1]
    
    def __rshift__(self, other):
        new_stages = [stage for stage in self.stages]
        if isinstance(other, StagedEstimator):
            new_stages += other.stages
        else:
            new_stages += [other]
        return StagedEstimator(new_stages)
            
    def __rrshift__(self, other):
        new_stages = self.stages.copy()
        if isinstance(other, StagedEstimator):
            new_stages = other.stages + new_stages
        else:
            new_stages = [other] + new_stages
        return StagedEstimator(new_stages)
    
    def _transform_args(self, data):
        result = {'X': data['X']}
        if 'exposure' in data:
            result['exposure'] = data['exposure']
        return result
    
    def _update(self, data):
        for stage in self.intermediate_stages_:
            try:
                # Stage knows to discard whatever it doesn't need
                stage.update(data)
            except AttributeError:
                data['X'] = safe_call(stage.transform, self._transform_args(data))
    
    def _sym_update(self):
        expressions = None
        for stage in self.intermediate_stages_:
            stage_expressions = sym_update(stage)
            if expressions is not None:
                new_expressions = []
                inputs = syms(stage)
                for expr in stage_expressions:
                    assert not set(inputs) & expr.free_symbols, 'Name collision in stage symbols'
                    new_expr = expr
                    for var, input_expr in zip(inputs, expressions):
                        new_expressions.append(new_expr.subs(var, input_expr))
                expressions = new_expressions
            else:
                expressions = stage_expressions
        return expressions
 
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight, exposure=exposure)
        self.intermediate_stages_ = []
        for stage in self.intermediate_stages:
            # Stage knows to discard whatever it doesn't need
            stage_ = clone(stage)
            safe_call(stage_.fit, data)
            try:
                stage_.update(data)
            except AttributeError:
                try:
                    data['X'] = safe_call(stage_.transform, self._transform_args(data))
                except:
                    data['X'] = safe_call(stage_.transform, self._transform_args(data))
            self.intermediate_stages_.append(stage_)
#             if 'X' in data and len(data['X'].shape) == 1:
#                 data['X'] = data['X'][:, None]
        self.final_stage_ = safe_call(clone(self.final_stage).fit, data)
        return self
    
    def transform(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.transform, data)
    
    def sym_predict(self):
        expressions = self._sym_update()
        expression = sym_predict(self.final_stage_)
        symbols =  syms(self.final_stage_)
        for expr, sym in zip(expressions, symbols):
            expression = expression.subs(sym, expr)
        return expression
    
    def syms(self):
        return syms(self.intermediate_stages_[0])
    
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.predict, data)
        
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.predict_proba, data)
    
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.predict_log_proba, data)
    
    def score(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.score, data)
    
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        self._update(data)
        return safe_call(self.final_stage_.decision_function, data)

def staged(estimator):
    return StagedEstimator([estimator])
        
# def as_pipeline(estimator):
#     try:
#         return estimator.as_pipeline()
#     except AttributeError:
#         return STPipeline([(name_estimator(estimator), estimator)])

class STSimpleEstimator(STEstimator):
    
    def __rshift__(self, other):
        '''
        self >> other
        '''
        return staged(self) >> other
        
    def __rrshift__(self, other):
        '''
        other >> self
        '''
        return staged(other) >> self
        
    def __lshift__(self, other):
        '''
        self << other
        '''
        return staged(other) >> self
        
    def __rlshift__(self, other):
        '''
        other << self
        '''
        return staged(self) >> other

#     def as_pipeline(self):
#         return STPipeline([(name_estimator(self), self)])
        
# class STPipeline(STEstimator, Pipeline):
#     def as_pipeline(self):
#         return self
#     
#     def __rshift__(self, other):
#         '''
#         self >> other
#         '''
#         other = as_pipeline(other)
#         steps = self.steps + other.steps
#         names = [step[0] for step in steps]
#         estimators = [step[1] for step in steps]
#         return STPipeline(zip(combine_named_estimators(names), estimators))

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
            spec = getargspec(method)
            def out(*args, **kwargs):
                setattr(obj, clone_name, method(*args, **kwargs))
                return obj
            out._spec = spec
        else:
            delegate = getattr(obj, clone_name)
            method = getattr(delegate, self.method_name)
            spec = getargspec(method)
            def out(*args, **kwargs):
                return method(*args, **kwargs)
            out._spec = spec
#             out = lambda *args, **kwargs: method(*args, **kwargs)
#             out = method
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
            self._delegates = {}
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
                  'predict_log_proba': 'estimator', 'transform': 'estimator'}
    def __init__(self, estimator):
        self.estimator = estimator

# class EstimatorStage(DelegatingEstimator):
#     def __init__(self, estimator, method_args):
#         self.estimator = estimator
#         self._create_delegates('estimator', standard_methods)
#         
#     def update(self, data):
#         '''
#         Update data in place to pass to the next stage in a pipeline.
#         '''
# class TransformerStage(EstimatorStage):
#     def update(self, data):
#         data['X'] = self.transform(**data)
        
# class ConcatenatingResponseStage(EstimatorStage):
#     def update(self, data):
#         data[''] = self.transform(**data)
class Wrapper(object):
    def __init__(self, content):
        self.content = content

class AlreadyFittedEstimator(DelegatingEstimator):
    def __init__(self, estimator):
        if isinstance(estimator, Wrapper):
            self.estimator = estimator
        else:
            self.estimator = Wrapper(estimator)
#         self.estimator_ = self.estimator.content
        self._create_delegates('estimator', non_fit_methods)
    
    @property
    def estimator_(self):
        return self.estimator.content
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        return self
    
class BoundedEstimator(DelegatingEstimator):
    def __init__(self, estimator, lower_bound=float('-inf'), upper_bound=float('inf')):
        self.estimator = estimator
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Delegate everything except predict and score
        self._create_delegates('estimator', ['fit', 'predict_proba', 'decision_function', 
                         'predict_log_proba', 'transform'])
    
    @if_delegate_has_method('estimator')
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        raw_prediction = self.estimator_.predict(**data)
        bounded_prediction = np.maximum(np.minimum(raw_prediction, self.upper_bound), self.lower_bound)
        return bounded_prediction
        
    @if_delegate_has_method('estimator')
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_proba(**data)
    
    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_log_proba(**data)
    
    @if_delegate_has_method('estimator')
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.decision_function(**data)
#     def predict(self):
#     
# def mask_estimator(estimator, mask):
#     return MaskedEstimator(estimator, mask)
#     try:
#         return estimator.mask(mask)
#     except AttributeError:
#         return MultipleResponseEstimator([(name_estimator(estimator), mask, estimator)])

# def is_masked(estimator):
#     return isinstance(estimator, MultipleResponseEstimator)

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

def safe_col_select(data, cols):
    if hasattr(data, 'loc'):
        return data.loc[:, cols]
    else:
        return data[:, cols]
        

class BaseRowSubsetTransformer(STSimpleEstimator):
    '''
    Removes some rows for whatever reason.
    '''
    def fit(self, X=None, y=None, sample_weight=None, exposure=None):
        return self
    
    def transform(self, X=None, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight, 
                                  exposure=exposure)
        rows = self._predicate(data)
        return _subset(X, rows)
    
    def update(self, data):
        rows = self._predicate(data)
        for k in data.keys():
            data[k] = _subset(data[k], rows)

class BaseRowSubsetFitter(DelegatingEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self._create_delegates('estimator', standard_methods)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight, 
                                  exposure=exposure)
        self.estimator_ = clone(self.estimator).fit(**(_subset_data(data, self._predicate(data))))
        return self

class ArgumentFixingEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, arg_dict):
        self.estimator = estimator
        self.arg_dict = arg_dict
#         self._create_delegates('estimator', non_fit_methods)
    
    @if_delegate_has_method('estimator')
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                  exposure=exposure)
        if 'fit' in self.arg_dict:
            data.update(self.arg_dict['fit'])
        self.estimator_ = clone(self.estimator).fit(**data)
        return self
    
    @if_delegate_has_method('estimator')
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'predict' in self.arg_dict:
            data.update(self.arg_dict['predict'])
        return self.estimator_.predict(**data)
    
    @if_delegate_has_method('estimator')
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'predict_proba' in self.arg_dict:
            data.update(self.arg_dict['predict_proba'])
        return self.estimator_.predict_proba(**data)
    
    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'predict_log_proba' in self.arg_dict:
            data.update(self.arg_dict['predict_log_proba'])
        return self.estimator_.predict_log_proba(**data)
    
    @if_delegate_has_method('estimator')
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        if 'decision_function' in self.arg_dict:
            data.update(self.arg_dict['decision_function'])
        return self.estimator_.decision_function(**data)

def non_null_rows(arr):
    if hasattr(arr, 'notnull'):
        return arr.notnull().any(axis=1)
    else:
        return ~(np.isnan(arr).any(axis=1))

def non_null_rows_dict(data):
    result = None
    for v in data.values():
        if result is None:
            result = np.ones(shape=v.shape[0], dtype=bool)
        result &= non_null_rows(v)
    if result is None:
        result = slice(None)
    return result

class NonMissingRowSubsetMixin(object):
    def _predicate(self, data):
        return non_null_rows_dict(data)
    
class NonNullSubsetFitter(BaseRowSubsetFitter, NonMissingRowSubsetMixin):
    pass
    
class ColumnSubsetTransformer(STSimpleEstimator):
    '''
    Takes all data from X and splits it into X, y, sample_weight, and exposure.  Use with 
    StagedEstimator.  If used as transformer, only gives X (with appropriate subset of columns).
    '''
    def __init__(self, x_cols=slice(None), y_cols=None,
                  sample_weight_cols=None, exposure_cols=None):
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.sample_weight_cols = sample_weight_cols
        self.exposure_cols = exposure_cols
        
    def fit(self, X=None, y=None, sample_weight=None, exposure=None):
        return self
    
    def transform(self, X=None, y=None, sample_weight=None, exposure=None):
        return safe_col_select(X, self.x_cols)
    
    def update(self, args):
        keys = {'X':self.x_cols, 'y':self.y_cols, 'sample_weight':self.sample_weight_cols, 
                'exposure':self.exposure_cols}
        X = args['X']
        for key, cols in keys.items():
            if cols is not None:
                try:
                    args[key] = safe_col_select(X, cols)
                except KeyError:
                    if key in {'X', 'exposure'}:
                        raise
                    else:
                        pass
    
class MaskedEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, mask):
        self.estimator = estimator
        self.mask = convert_mask(mask)
    
    def _mask_y(self, y):
        if y is None:
            return None
        result = y[:, self.mask]
        if len(result) == 0:
            return None
        return result
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=self._mask_y(y), sample_weight=sample_weight,
                                  exposure=exposure)
        if len(data['y'].shape) > 1 and data['y'].shape[1] == 1:
            data['y'] = np.ravel(data['y'])
        self.estimator_ = clone(self.estimator).fit(**data)
        return self
        
    @if_delegate_has_method('estimator')
    def score(self, X, y=None, sample_weight=None, exposure=None):
        data = self._process_args(X=X, y=self._mask_y(y), sample_weight=sample_weight,
                                  exposure=exposure)
        return self.estimator_.score(**data)
    
    @if_delegate_has_method('estimator')
    def predict(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict(**data)
    
    @if_delegate_has_method('estimator')
    def predict_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_proba(**data)
    
    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.predict_log_proba(**data)
    
    @if_delegate_has_method('estimator')
    def decision_function(self, X, exposure=None):
        data = self._process_args(X=X, exposure=exposure)
        return self.estimator_.decision_function(**data)
        
class MultiEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimators):
        self.estimators = estimators
    
    def __and__(self, other):
        new_estimators = [est for est in self.estimators]
        if isinstance(other, MultiEstimator):
            new_estimators += other.estimators
        else:
            new_estimators += [other]
        return MultiEstimator(new_estimators)
            
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                  exposure=exposure)
        self.estimators_ = []
        for estimator in self.estimators:
            self.estimators_.append(clone(estimator).fit(**args))
        return self
    
    def transform(self, X, y=None, sample_weight=None, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.transform(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def predict(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def predict_proba(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict_proba(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def predict_log_proba(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict_log_proba(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def decision_function(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.decision_function(**args)
            if len(result.shape) == 1:
                result = result[:, None]
            results.append(result)
        return np.concatenate(results, axis=1)
    
    def score(self, X, y=None, sample_weight=None, exposure=None):
        args = self._process_args(X=X, y=y, sample_weight=sample_weight, exposure=exposure)
        results = []
        for estimator in self.estimators_:
            result = estimator.predict_log_proba(**args)
            results.append(result)
        return np.array(results, axis=1)
    
    
# class UnionEstimator(STSimpleEstimator, MetaEstimatorMixin):
#     def __init__(self, estimators):
#         masks = [convert_mask(est[0]) for est in estimators]
#         if not compatible_masks(masks):
#             raise ValueError('Masks do not have compatible sizes')
#         estimators_ = [est[1] for est in estimators]
#         self.estimators = zip(masks, estimators_)
#     
#     def fit(self, X, y=None, sample_weight=None, exposure=None):
#         args = self._process_args(X=X, y=y, sample_weight=sample_weight,
#                                   exposure=exposure)
#         
    
#     
# class MultipleResponseEstimator(STSimpleEstimator, MetaEstimatorMixin):
#     def __init__(self, estimators):
#         masks = [convert_mask(est[0]) for est in estimators]
#         if not compatible_masks(masks):
#             raise ValueError('Masks do not have compatible sizes')
#         estimators_ = [est[1] for est in estimators]
#         self.estimators = zip(masks, estimators_)
#     
#     def __and__(self, other):
#         other = MultipleResponseEstimator(other, slice(None))
#         all_estimators = self.estimators + other.estimators
#         masks = [est[0] for est in all_estimators]
#         estimators = [est[1] for est in all_estimators]
#         if not compatible_masks(masks):
#             raise ValueError('Masks do not have compatible sizes')
#         return MultipleResponseEstimator(zip(masks, estimators))
#         
#     def mask(self, mask):
#         assert mask == slice(None)
#         return self
#     
#     def _masks(self):
#         return [mask for _, mask, _ in self.estimators]
#     
#     @property
#     def _estimator_type(self):
#         if all([is_classifier(estimator) for estimator in self.estimators.values()]):
#             return 'classifier'
#         elif all([is_regressor(estimator) for estimator in self.estimators.values()]):
#             return 'regressor'
#         else:
#             return 'mixed'
#     
#     def fit(self, X, y, sample_weight=None, exposure=None):
#         fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
#                                       exposure=exposure)
#         self.estimators_ = []
#         for columns, model in self.estimators:
#             dargs = kwargs.copy()
#             dargs.update(fit_args.get(name, {}))
#             
#             # Select the appropriate columns
#             y_ = y[:, columns]
#             if y_.shape[1] == 1:
#                 y_ = y_[:, 0]
#             
#             # Fit the estimator
#             self.estimators_.append((name, columns, clone(model).fit(X, y_, *args, **dargs)))
#         self.estimators_dict_ = {name: (columns, model) for name, columns, model in self.estimators_}
#         
#         # Do a prediction on a single row of data for each estimator in order to 
#         # determine the number of predicted columns for each one
#         X_ = X[0:1, :]
#         self.prediction_columns_ = []
#         for name, columns, model in self.estimators_:
#             prediction = model.predict(X_)
#             if len(prediction.shape) > 1:
#                 n_columns = prediction.shape[1]
#             else:
#                 n_columns = 1
#             self.prediction_columns_ += [name] * n_columns
#         
#         return self
#     
#     def predict(self, X, predict_args=None, *args, **kwargs):
#         if predict_args is None:
#             predict_args = {}
#         predictions = []
#         for name, columns, model in self.estimators_:  # @UnusedVariable
#             dargs = kwargs.copy()
#             dargs.update(predict_args.get(name, {}))
#             prediction = model.predict(X, *args, **dargs)
#             predictions.append(prediction if len(prediction.shape) == 2 else prediction[:, None])
#         return np.concatenate(predictions, axis=1)
#     
#     def predict_proba(self, X, predict_args=None, *args, **kwargs):
#         if predict_args is None:
#             predict_args = {}
#         predictions = []
#         for name, columns, model in self.estimators_:  # @UnusedVariable
#             dargs = kwargs.copy()
#             dargs.update(predict_args.get(name, {}))
#             prediction = model.predict_proba(X, *args, **dargs)
#             predictions.append(prediction if len(prediction.shape) == 2 else prediction[:, None])
#         return np.concatenate(predictions, axis=1)



