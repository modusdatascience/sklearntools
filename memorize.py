from sklearntools.sklearntools import DelegatingEstimator
from sklearn.base import clone
from toolz.functoolz import curry


class MemorizingEstimator(DelegatingEstimator):
    '''
    Create an estimator that will remember previous fits with the same data and load
    from disk to avoid re-fitting if an identical fit has been done before.
    
    Parameters
    ==========
    
    memory (joblib.Memory): A Memory object that will be used for memoization of 
        previously fitted models.
    
    estimator (estimator): An estimator to fit.  If this estimator has been 
        used before with the same memory settings, a previous fit may be loaded from 
        disk if fitting is attempted with identical data.
    
    Attributes
    ==========
    
    estimator_ (estimator): The fitted estimator, which may have been loaded from disk.
    
    '''
    def __init__(self, memory, estimator):
        self.memory = memory
        self.estimator = estimator
        self._fit = self.memory.cache(self._fit)
        self._fit_transform = self.memory.cache(self._fit_transform)
        self._fit_predict = self.memory.cache(self._fit_predict)
        self._create_delegates('estimator', ['predict', 'transform', 'predict_proba', 
                                             'predict_log_proba', 'decision_function'])
        
    
    def _fit(self, estimator, fit_args):
        self.loaded_from_cache_ = False
        return clone(estimator).fit(**fit_args)
    
    def _fit_transform(self, estimator, fit_args):
        self.loaded_from_cache_ = False
        model = clone(estimator)
        transformed = model.fit_transform(**fit_args)
        return model, transformed
    
    def _fit_predict(self, estimator, fit_args):
        self.loaded_from_cache_ = False
        model = clone(estimator)
        predicted = model.fit_predict(**fit_args)
        return model, predicted
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        self.loaded_from_cache_ = True
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        self.estimator_ = self._fit(self.estimator, fit_args)
        return self
    
    def fit_transform(self, X, y=None, sample_weight=None, exposure=None):
        self.loaded_from_cache_ = True
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        self.estimator_, result = self._fit_transform(self.estimator, fit_args)
        return result
    
    def fit_predict(self, X, y=None, sample_weight=None, exposure=None):
        self.loaded_from_cache_ = True
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        self.estimator_, result = self._fit_predict(self.estimator, fit_args)
        return result


@curry
def memorize(memory, estimator):
    return MemorizingEstimator(memory, estimator)

