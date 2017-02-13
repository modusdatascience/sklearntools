from sklearntools import STSimpleEstimator, DelegatingEstimator, non_fit_methods,\
    standard_methods, _subset, safe_col_select, safe_call
from sklearn.base import MetaEstimatorMixin, is_classifier, clone,\
    TransformerMixin
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib.parallel import Parallel, delayed
import numpy as np
from sklearn.utils.metaestimators import if_delegate_has_method
from sym import sym_transform, sym_predict, sym_predict_proba, syms, sym_predict_parts
from sklearntools import safe_assign_subset, _fit_and_predict
from sympy.core.symbol import Symbol

# def _fit_and_predict(estimator, X, y, train, test, sample_weight=None, exposure=None):
#     '''
#     Fits on the train set and predicts on the test set.
#     '''
#     fit_args = {'X': _subset(X, train)}
#     if y is not None:
#         fit_args['y'] = _subset(y, train)
#     if sample_weight is not None:
#         fit_args['sample_weight'] = _subset(sample_weight, train)
#     if exposure is not None:
#         fit_args['exposure'] = _subset(exposure, train)
#     estimator.fit(**fit_args)
#     
#     predict_args = {'X': _subset(X, test)}
#     if exposure is not None:
#         predict_args['exposure'] = _subset(exposure, test)
#     prediction = estimator.predict(**predict_args)
#     
#     return estimator, prediction

class ThresholdClassifier(STSimpleEstimator, MetaEstimatorMixin):
    _estimator_type = 'classifier'
    def __init__(self, classifier, threshold=0, multiplier=1):
        self.classifier = classifier
        self.threshold = threshold
        self.multiplier = multiplier
    
    def fit(self, X, y, sample_weight=None, exposure=None):
        clas_args = {'X': X, 'y': self.multiplier * y > self.threshold}
        if sample_weight is not None:
            clas_args['sample_weight'] = sample_weight
        if exposure is not None:
            clas_args['exposure'] = exposure
        self.classifier_ = safe_call(clone(self.classifier).fit, clas_args)
        return self
    
    def predict(self, X, exposure=None):
        clas_args = {'X': X}
        if exposure is not None:
            clas_args['exposure'] = exposure
        return safe_call(self.classifier_.predict, clas_args)
    
    @if_delegate_has_method(delegate='classifier')
    def decision_function(self, X, exposure=None):
        clas_args = {'X': X}
        if exposure is not None:
            clas_args['exposure'] = exposure
        return safe_call(self.classifier_.decision_function, clas_args)
    
    @if_delegate_has_method(delegate='classifier')
    def predict_proba(self, X, exposure=None):
        clas_args = {'X': X}
        if exposure is not None:
            clas_args['exposure'] = exposure
        return safe_call(self.classifier_.predict_proba, clas_args)
    
    def sym_predict(self):
        return sym_predict(self.classifier_)
    
    def syms(self):
        return syms(self.classifier_)
    
    def sym_predict_proba(self):
        return sym_predict_proba(self.classifier_)
    
    @if_delegate_has_method(delegate='classifier')
    def predict_log_proba(self, X, exposure=None):
        clas_args = {'X': X}
        if exposure is not None:
            clas_args['exposure'] = exposure
        return safe_call(self.classifier_.predict_log_proba, clas_args)

# class SeparateSortingEstimator(STSimpleEstimator, MetaEstimatorMixin):
#     '''
#     Sorts X and y separately before fitting.  This is only useful if you really
#     know what you're doing.
#     '''
#     def __init__(self, estimator):
#         self.estimator = estimator
#         
#     def fit(self, X, y, sample_weight=None, exposure=None):
#         pass

def moving_average(y, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    if len(y.shape) == 1:
        y = y[:,None]
    result = []
    for j in range(y.shape[1]):
        result.append(np.convolve(y[:, j], window, 'valid')[:, None])
    return np.concatenate(result, axis=1)

class MovingAverageSmoothingEstimator(DelegatingEstimator):
    '''
    The idea is that X is a prediction from some other estimator.  Otherwise, the order 
    of X is not very meaningful and the resultimg model will probably be garbage. 
    There could perhaps be other particular applications, but make sure you understand why 
    the order of X is meaningful before using this estimator.
    '''
    def __init__(self, estimator, window_size=25, sort_order=None, sort_algorithm='quicksort'):
        self.estimator = estimator
        self.window_size = window_size
        self.sort_order = sort_order
        self.sort_algorithm = sort_algorithm
        self._create_delegates('estimator', non_fit_methods)
        
    def sym_predict(self):
        return sym_predict(self.estimator_)
    
    def syms(self):
        return syms(self.estimator_)
        
    def fit(self, X, y):
#         if sample_weight is not None:
#             raise ValueError('sample_weight not supported')
#         if exposure is not None:
#             raise ValueError('exposure not supported')
        y = np.asarray(y)
        X = np.asarray(X)
        if len(y.shape) == 1:
            y = y[:, None]
        if len(X.shape) == 1:
            X = X[:, None]
            
        # Sort on X
        order = np.argsort(X, axis=0, kind=self.sort_algorithm, order=self.sort_order)[:,0]
        
        # Moving average on X and y based on sort order of X
        X_ = moving_average(X[order, :], self.window_size)
        try:
            y_ = moving_average(y[order, :], self.window_size)
        except:
            y_ = moving_average(y[order, :], self.window_size)
        # Fit estimator on the moving averages
        if y_.shape[1] == 1:
            y_ = np.ravel(y_)
        self.estimator_ = clone(self.estimator).fit(X_, y_)
        
        return self

class ConstantRateToTotalEstimator(STSimpleEstimator, MetaEstimatorMixin):
    '''
    When fit, expects to receive predicted rate of accumulation (of, say, cost) as 
    X and observed totals (say, total cost during exposure) as y.  Also expects
    to receive exposure.  Once fit, predicts total (based on exposure).
    
    Basically, all this does is multiply rate by exposure.  Recommend piping in a 
    PredictorTransformer.
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y, sample_weight=None, exposure=None):
        return self
    
    def predict(self, X, exposure):
        if exposure is None:
            raise ValueError('Must provide exposure')
        if len(X.shape) == 1:
            X = X[:, None]
        return exposure * X
    
    def transform(self, X, exposure):
        return self.predict(X, exposure)
    

class HazardToRiskEstimator(STSimpleEstimator, MetaEstimatorMixin):
    '''
    When fit, expects to receive predicted cumulative hazard per exposure as X and 
    observed event occurrence (boolean/binary) as y.  Also expects to receive exposure.  
    Once fit, predicts risk.
    
    Suggestion: Use this with a MovingAverageSmoothingEstimator as estimator and wrap it 
    in a ThresholdClassifier while pipelining in a PredictorTransformer.
    '''
    def __init__(self, estimator):
        self.estimator = estimator
    
    def _preprocess_x(self, X, exposure):
        if exposure is None:
            raise ValueError('Must provide exposure')
        if len(X.shape) == 1:
            X = X[:, None]
        if len(exposure.shape) == 1:
            exposure = exposure[:, None]
        return 1. - np.exp(-X * exposure)
    
    def fit(self, X, y, sample_weight=None, exposure=None):
        y = np.asarray(y)
        fit_args = {'X': self._preprocess_x(X, exposure),
                    'y': y[:,0] if len(y.shape) == 2 and y.shape[1] == 1 else y}
        if sample_weight is not None:
            fit_args['sample_weight'] = sample_weight
        
        self.estimator_ = clone(self.estimator).fit(**fit_args)
        return self
    
    def predict(self, X, exposure=None):
        return self.estimator_.predict(self._preprocess_x(X, exposure))
    
    def predict_proba(self, X, exposure=None):
        return self.estimator_.predict_proba(self._preprocess_x(X, exposure))
    
    def predict_log_proba(self, X, exposure=None):
        return self.estimator_.predict_log_proba(self._preprocess_x(X, exposure))
    
    def decision_function(self, X, exposure=None):
        return self.estimator_.decision_function(self._preprocess_x(X, exposure))

class PredictorTransformer(DelegatingEstimator):
    '''
    Just overrides transform to use predict.  Useful for pipelines.
    '''
    def __init__(self, estimator):
        self.estimator = estimator
        self._create_delegates('estimator', standard_methods)
    
    def transform(self, X, exposure=None):
        args = {'X': X}
        if exposure is not None:
            args['exposure'] = exposure
        result = self.predict(**args)
        if len(result.shape) == 1:
            result = result[:, None]
        return result
    
    def syms(self):
        return self.estimator_.syms()
    
    def sym_transform(self):
        return [self.sym_predict()]
    
    def sym_predict(self):
        return self.estimator_.sym_predict()
    
    def sym_transform_parts(self, target=None):
        print 'sym_transform_parts', self
        return sym_predict_parts(self, target)
    
class SelectorTransformer(STSimpleEstimator):
    '''
    Just grab some input columns and pass them through.
    '''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, sample_weight=None, exposure=None):
        return self
    
    def transform(self, X, exposure=None):
        return safe_col_select(X, self.columns)
    
    def syms(self):
        return [Symbol(col) for col in self.columns]
    
    def sym_transform(self):
        return self.syms()
    
def no_cv(X, y):
    yield np.ones(X.shape[0]).astype(bool), np.ones(X.shape[0]).astype(bool)

class CalibratedEstimatorCV(STSimpleEstimator, MetaEstimatorMixin):
    '''
    cv = 1 gives no cross validation.
    '''
    def __init__(self, estimator, calibrator, est_weight=True, est_exposure=False, 
                 cal_weight=True, cal_exposure=True, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.calibrator = calibrator
        self.est_weight = est_weight
        self.est_exposure = est_exposure
        self.cal_weight = cal_weight
        self.cal_exposure = cal_exposure
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
    
    def syms(self):
        return syms(self.estimator_)
    
    def sym_predict(self):
        est = sym_predict(self.estimator_)
        cal = sym_predict(self.calibrator_)
        cal_vars = syms(self.calibrator_)
        assert len(cal_vars) == 1
        return cal.subs(cal_vars[0], est)
    
    def sym_predict_parts(self, target=None):
        print 'sym_predict_parts', self
        parts = sym_predict_parts(self.estimator_, target)
        return sym_predict_parts(self.calibrator_, parts)
    
    def sym_transform(self):
        return sym_transform(self.estimator_)
    
    @property
    def _estimator_type(self):
        return self.calibrator._estimator_type
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        if self.cv == 1:
            cv = no_cv(X=X, y=y)
        else:
            if hasattr(self.cv, 'split'):
                cv = self.cv.split(X, y)
            else:
                cv = check_cv(self.cv, X=X, y=y, classifier=is_classifier(self.calibrator))
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        pre_dispatch=self.pre_dispatch)
        
        # Fit the estimator on each train set and predict with it on each test set
        fit_args = {'X': X}
        if y is not None:
            fit_args['y'] = y
        if self.est_weight and sample_weight is not None:
            fit_args['sample_weight'] = sample_weight
        if self.est_exposure and exposure is not None:
            fit_args['exposure'] = exposure
        
        # Do the cross validation fits
        cv_fits = parallel(delayed(_fit_and_predict)(clone(self.estimator), fit_args, train, test) for train, test in cv)
        
        # Combine predictions from cv fits
        prediction = np.empty_like(y)
        for fit in cv_fits:
            safe_assign_subset(prediction, fit[2], fit[1])
        
#         fit_predict_results = parallel(delayed(_fit_and_predict)(estimator=clone(self.estimator),
#                                        train=train, test=test, **fit_args) for train, test in cv)
#         
#         # Combine the predictions
#         prediction = np.empty_like(y)
#         for _, pred, _, test in zip(fit_predict_results, cv):
            
#         prediction = np.concatenate([pred for _, pred in cv_fits], axis=0)
        
        # Fit the calibrator on the predictions
        cal_args = {'X': prediction[:, None] if len(prediction.shape) == 1 else prediction, 
                    'y': y}
        if self.cal_weight and sample_weight is not None:
            cal_args['sample_weight'] = sample_weight
        if self.cal_exposure and exposure is not None:
            cal_args['exposure'] = exposure
        self.calibrator_ = clone(self.calibrator).fit(**cal_args)
        
        # Fit the estimator on the entire data set
        self.estimator_ = clone(self.estimator).fit(**fit_args)
        
        return self
    
    @if_delegate_has_method(delegate='estimator')
    def transform(self, X):
        return self.estimator_.transform(X)
    
    def _estimator_predict(self, X, exposure=None):
        est_args = {'X': X}
        if self.est_exposure and exposure is not None:
            est_args['exposure'] = exposure
        est_prediction = self.estimator_.predict(**est_args)
        if len(est_prediction.shape) == 1:
            est_prediction = est_prediction[:, None]
        return est_prediction
    
    def predict(self, X, exposure=None):
        est_prediction = self._estimator_predict(X, exposure)
        cal_args = {'X': est_prediction}
        if self.cal_exposure and exposure is not None:
            cal_args['exposure'] = exposure
        return self.calibrator_.predict(**cal_args)
    
    @if_delegate_has_method(delegate='calibrator')
    def decision_function(self, X, exposure=None):
        est_prediction = self._estimator_predict(X, exposure)
        cal_args = {'X': est_prediction[:, None] if len(est_prediction.shape) == 1 
                            else est_prediction}
        if self.cal_exposure and exposure is not None:
            cal_args['exposure'] = exposure
        return self.calibrator_.decision_function(**cal_args)

    @if_delegate_has_method(delegate='calibrator')
    def predict_proba(self, X, exposure=None):
        est_prediction = self._estimator_predict(X, exposure)
        cal_args = {'X': est_prediction}
        if self.cal_exposure and exposure is not None:
            cal_args['exposure'] = exposure
        return self.calibrator_.predict_proba(**cal_args)

    @if_delegate_has_method(delegate='calibrator')
    def predict_log_proba(self, X, exposure=None):
        est_prediction = self._estimator_predict(X, exposure)
        cal_args = {'X': est_prediction}
        if self.cal_exposure and exposure is not None:
            cal_args['exposure'] = exposure
        return self.calibrator_.predict_log_proba(**cal_args)

class IdentityTransformer(STSimpleEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None, sample_weight=None):
        return self
     
    def transform(self, X, y=None):
        return X

class LogTransformer(STSimpleEstimator, TransformerMixin):
    def __init__(self, offset = 1.):
        self.offset = offset
     
    def fit(self, X, y=None, sample_weight=None):
        return self
     
    def transform(self, X, y=None):
        return np.log(self.offset + X)

def index(table, idx0, idx1):
    if hasattr(table, 'loc'):
        return table.loc[idx0, idx1]
    else:
        return table[idx0, idx1]

class MultiTransformer(STSimpleEstimator, TransformerMixin, MetaEstimatorMixin):
    def __init__(self, transformers):
        self.transformers = transformers
        
    def fit(self, X, y=None, sample_weight=None):
        self.transformers_ = [(name, mask, clone(trans)) for name, mask, trans in self.transformers]
        for _, mask, trans in self.transformers_:
            args = {'X': index(X, slice(None), mask)}
            if y is not None:
                args['y'] = y
            if sample_weight is not None:
                args['sample_weight'] = sample_weight
            trans.fit(**args)
        return self
    
    def transform(self, X):
        results = []
        for _, mask, trans in self.transformers_:
            results.append(trans.transform(index(X, slice(None), mask)))
        return np.concatenate(results, axis=1)

class ProbaPredictingEstimator(DelegatingEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, y, *args, **kwargs):
        self.estimator_ = clone(self.estimator)
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = np.ravel(y)
        self.estimator_.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict_proba(X, *args, **kwargs)[:, 1:]
    
    def sym_transform(self):
        return sym_transform(self.estimator_)
    
    def sym_predict(self):
        return sym_predict_proba(self.estimator_)
    
    def syms(self):
        return syms(self.estimator_)
        
class ResponseTransformingEstimator(DelegatingEstimator):
    def __init__(self, estimator, transformer, est_weight=False, est_exposure=False, trans_weight=False,
                 trans_exposure=False):
        self.estimator = estimator
        self.transformer = transformer
        self.est_weight = est_weight
        self.est_exposure = est_exposure
        self.trans_weight = trans_weight
        self.trans_exposure = trans_exposure
        self._create_delegates('estimator', ['predict', 'transform', 'predict_proba', 
                                             'predict_log_proba', 'decision_function'])
    def syms(self):
        return syms(self.estimator_)
    
    def sym_predict(self):
        return sym_predict(self.estimator_)
    
    def sym_transform(self):
        return sym_transform(self.estimator_)
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
    def fit(self, X, y, sample_weight=None, exposure=None):
        transformer_args_ = {'X': y}
        if self.trans_weight:
            transformer_args_['sample_weight'] = sample_weight
        if self.trans_exposure:
            transformer_args_['exposure'] = exposure
        self.transformer_ = clone(self.transformer).fit(**transformer_args_)
        y_transformed = self.transformer_.transform(y)
        estimator_args_ = {'X': X, 'y': y_transformed}
        if self.est_weight:
            estimator_args_['sample_weight'] = sample_weight
        if self.est_exposure:
            estimator_args_['exposure'] = exposure
        self.estimator_ = clone(self.estimator).fit(**estimator_args_)
        return self
    
    def score(self, X, y, sample_weight=None, exposure=None):
        y_transformed = self.transformer_.transform(y)
        args = {'X': X, 'y': y_transformed}
        if self.est_weight:
            args['sample_weight'] = sample_weight
        if self.est_exposure:
            args['exposure'] = exposure
        return self.estimator_.score(**args)
#     
#     def predict(self, X):
#         return self.estimator_.predict(X)
#     
#     @if_delegate_has_method('estimator')
#     def transform(self, X):
#         return self.estimator_.transform(X)
#     
#     @if_delegate_has_method('estimator')
#     def predict_proba(self, X):
#         return self.estimator_.predict_proba(X)
#     
#     @if_delegate_has_method('estimator')
#     def predict_log_proba(self, X):
#         return self.estimator_.predict_log_proba(X)
#     
#     @if_delegate_has_method('estimator')
#     def decision_function(self, X):
#         return self.estimator_.decision_function(X)
