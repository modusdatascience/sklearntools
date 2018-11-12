from sklearn.externals.joblib.parallel import Parallel, delayed
from .calibration import no_cv
from sklearn.cross_validation import check_cv
from sklearn.base import is_classifier, clone
from .sklearntools import _fit_and_predict, non_fit_methods, BaseDelegatingEstimator, safe_assign_subset, safer_call
import numpy as np
# from .sym.sym_predict import sym_predict
# from .sym.syms import syms
# from .sym.sym_predict_parts import sym_predict_parts
# from .sym.sym_transform_parts import sym_transform_parts
from .sklearntools import shrinkd
# from .sym.input_size import input_size
from sklearn2code.sym.base import sym_predict as s2c_sym_predict
from sklearn.model_selection._split import StratifiedKFold, BaseCrossValidator,\
    KFold
from _collections_abc import Iterable
from six import with_metaclass
from abc import ABCMeta, abstractmethod
from toolz.curried import valmap
from sklearntools.sklearntools import safe_column_select

class CrossValidatingEstimator(BaseDelegatingEstimator):
    def __init__(self, estimator, metric=None, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.metric = metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self._create_delegates('estimator', non_fit_methods)
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
#     def sym_predict(self):
#         return sym_predict(self.estimator_)
#     
#     def sym_predict_parts(self, target=None):
#         return sym_predict_parts(self.estimator_, target)
#     
#     def sym_transform_parts(self, target=None):
#         return sym_transform_parts(self.estimator_, target)
#     
#     def input_size(self):
#         return input_size(self.estimator_)
#     
#     def syms(self):
#         return syms(self.estimator_)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        # For later
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        pre_dispatch=self.pre_dispatch,
                        max_nbytes=None)
        
        # Extract arguments
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        
        # Sort out cv parameters
        if self.cv == 1:
            cv = no_cv(X=X, y=y)
        else:
            if hasattr(self.cv, 'split'):
                cv_args = dict(X=X)
                if y is not None:
                    cv_args['y'] = np.ravel(y)
                cv = self.cv.split(**cv_args)
            else:
                cv_args = dict(X=X)
                if y is not None:
                    cv_args['y'] = shrinkd(1,np.asarray(y))
                cv = check_cv(self.cv, classifier=is_classifier(self.estimator), **cv_args)
                
        # Do the cross validation fits
#         print(valmap(lambda x: x.shape, fit_args))
#         print('num_folds = %d' % self.cv.get_n_splits(X=X))
        cv_fits = parallel(delayed(_fit_and_predict)(clone(self.estimator), fit_args, train, test, self.verbose) for train, test in cv)
        
        # Combine predictions from cv fits
        prediction = np.empty_like(y) if y is not None else np.empty(shape=X.shape[0])
        for fit in cv_fits:
            safe_assign_subset(prediction, fit[2], fit[1])
        
        # Store cross validation models
        self.cv_estimators_ = [fit[0] for fit in cv_fits]
        self.cv_indices_ = [fit[2] for fit in cv_fits]
        self.cv_predictions_ = prediction
        
        # If a metric was provided, compute the score
        if self.metric is not None:
            metric_args = {}
            if 'sample_weight' in fit_args:
                metric_args['sample_weight'] = fit_args['sample_weight']
            if 'exposure' in fit_args:
                metric_args['exposure'] = fit_args['exposure']
            self.score_ = safer_call(self.metric, y, self.cv_predictions_, **metric_args)
        
        # Fit on entire data set
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(**fit_args)
        return self
        
    def fit_predict(self, X, y=None, sample_weight=None, exposure=None):
        self.fit(X=X, y=y, sample_weight=sample_weight, exposure=exposure)
        return self.cv_predictions_.copy()

@s2c_sym_predict.register(CrossValidatingEstimator)
def s2c_sym_predict_cross_validating_estimator(estimator):
    return s2c_sym_predict(estimator.estimator_)

class ThresholdStratifiedKFold(object):
    def __init__(self, thresholds, *args, column=None, **kwargs):
        self.column = column
        if isinstance(thresholds, Iterable):
            self.thresholds = list(thresholds)
        else:
            self.thresholds = [thresholds]
        self.stratified = StratifiedKFold(*args, **kwargs)
    
    def get_n_splits(self, *args,  **kwargs):
        return self.stratified.get_n_splits(*args, **kwargs)
    
    def split(self, X, y=None):
        if self.column is None:
            if y is None:
                raise ValueError('y argument required if column is None.')
            col = y
        else:
            col = safe_column_select(X, self.column)
        col_thresh = np.zeros(col.shape)
        for thresh in self.thresholds:
            col_thresh += col >= thresh
        for train, test in self.stratified.split(X, col_thresh):
            yield train, test
        
class HybridCV(with_metaclass(ABCMeta, BaseCrossValidator)):
    @abstractmethod
    def choose_loo(self, y):
        pass
    
    def __init__(self, n_folds, shuffle=True, **kwargs):
        self.n_folds = n_folds
        self.base_kfold = KFold(self.n_folds, shuffle=shuffle, **kwargs)
    
    @abstractmethod
    def get_n_splits(self, X, y, groups):
        pass
    
    def _iter_test_masks(self, X=None, y=None, groups=None):
        loo_indices = self.choose_loo(X, y, groups)
        not_loo_mask = np.ones(X.shape[0], dtype=bool)
        for idx in loo_indices:
            result = np.zeros(X.shape[0], dtype=bool)
            result[idx] = True
            not_loo_mask[idx] = False
            yield result
        for result in self.base_kfold._iter_test_masks(X=X, y=y, groups=groups):
            result_ = (result & not_loo_mask)
            if np.any(result_!=0):
                return result_
    
def all_false_predicate(X, y=None, groups=None):
    return np.zeros(shape=X.shape[0], dtype=bool)

class column_interval_predicate(object):
    def __init__(self, colname, lower=-np.inf, upper=np.inf):
        self.colname = colname
        self.lower = lower
        self.upper = upper
    
    def __call__(self, X, y=None, groups=None):
        z = np.ravel(X[self.colname])
        return (z > self.upper) | (z < self.lower)

# def column_interval_predicate(colname, lower=-np.inf, upper=np.inf):
#     def _column_interval_predicate(X, y=None, groups=None):
#         z = np.ravel(X[colname])
#         return (z > upper) | (z < lower)
#     return _column_interval_predicate
        
class PredicateHybridCV(HybridCV):
    def __init__(self, n_folds, shuffle=True, predicate=all_false_predicate, **kwargs):
        HybridCV.__init__(self, n_folds=n_folds, shuffle=shuffle, **kwargs)
        self.predicate=predicate
    
    def get_n_splits(self, X, y=None, groups=None):
        return np.sum(self.predicate(X, y, groups)) + self.n_folds
    
    def choose_loo(self, X, y=None, groups=None):
        return np.where(self.predicate(X, y, groups))[0]
        
class ThresholdHybridCV(HybridCV):
    def __init__(self, n_folds, shuffle=True, lower=-np.inf, upper=np.inf, **kwargs):
        HybridCV.__init__(self, n_folds=n_folds, shuffle=shuffle, **kwargs)
        self.lower = lower
        self.upper = upper
    
    def get_n_splits(self, X, y, groups=None):
        return np.sum((y > self.upper) | (y < self.lower)) + self.n_folds
    
    def choose_loo(self, X, y, groups):
        return np.where((y > self.upper) | (y < self.lower))[0]
    
    
    



