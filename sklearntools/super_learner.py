from sklearntools.sklearntools import STSimpleEstimator, growd, shrinkd
from sklearn2code.utility import tupify
from sklearn.base import clone
from sklearn.externals.joblib.parallel import Parallel
from sklearntools.kfold import CrossValidatingEstimator
import numpy as np
from toolz.dicttoolz import assoc, get_in, keyfilter, valmap, dissoc
from operator import __contains__
from toolz.functoolz import curry
from sklearn2code.sym.base import sym_predict, syms, sym_transform
from sklearn2code.sym.function import VariableFactory, Function, cart
from frozendict import frozendict
from collections import OrderedDict
from toolz.itertoolz import first
import scipy.optimize
from sklearn.exceptions import NotFittedError
from memorize import MemorizingEstimator, memorize
import pandas

class NonNegativeLeastSquaresRegressor(STSimpleEstimator):
    def __init__(self, normalize_coefs=True):
        '''
        normalize_coefs (bool): If True, coefficients will be normalized
            to add up to 1.
        '''
        self.normalize_coefs = normalize_coefs
    
    def fit(self, X, y):
        X = np.asarray_chkfinite(X)
        y = shrinkd(1, np.asarray_chkfinite(y))
        self.coefs_ = scipy.optimize.nnls(X, y)[0]
        if self.normalize_coefs:
            self.coefs_ /= np.sum(self.coefs_)
        print('self.coefs_ = {}'.format(self.coefs_))
        return self
    
    def predict(self, X):
        if not hasattr(self, 'coefs_'):
            raise NotFittedError()
        return np.dot(X, self.coefs_)
        
def sort_rows_independently(arr, inplace=True):
    '''
    Sort each row of the 2d array arr.
    '''
    if not inplace:
        arr = arr.copy()
    for i in range(arr.shape[0]):
        arr[i,:] = np.sort(arr[i,:])
    return arr if not inplace else None

class OrderTransformer(STSimpleEstimator):
    '''
    Transform input data into row-wise order statistics.  That is, 
    sort each row of the input matrix.
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        return self
    
    def transform(self, X):
        result = sort_rows_independently(X, inplace=False)
        result = pandas.DataFrame(result, columns=['O(%d/%d)'%(i+1, X.shape[1]) for i in range(X.shape[1])])
        return result
    
    
class SuperLearner(STSimpleEstimator):
    def __init__(self, regressors, meta_regressor, y_transformer=None, memory=None, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        '''
        
        Parameters
        ==========
        
        regressors (dict-like structure or items-like): The candidate models.
        
        meta_regressor (estimator): The meta-model that will combine the candidate models.
        
        y_transformer (transformer or NoneType): If not None, a transformer that will be used to 
            determine y when fitting the meta_regressor.
            
        memory (joblib.Memory or NoneType): If not None, used to cache fitted regressors so that they needn't be 
            re-fit in subsequent fits.  Useful when iteratively making changes to the SuperLearner that would
            require refitting.
            
        cv (int, cross-validation generator or an iterable): Passed to internal CrossValidatingEstimators to control
            cross-validation of the candidate models.
            
        n_jobs (int): Passed to the internal CrossValidatingEstimators to control number of jobs used in cross-
            validation of the candidate models.
            
        verbose (int): Level of verbosity.
        
        pre_dispatch (str): Passed to internal CrossValidatingEstimators.
        
        Attributes
        ==========
        cross_validating_estimators_ (OrderedDict): The fitted candidate models, wrapped in CrossValidatingEstimators and 
            possibly MemorizingEstimators (if memory is not None).
        
        y_transformer_ (transformer or NoneType): The fitted y_transformer or None.
        
        meta_regressor_ (estimator): The fitted meta-regressor.
        
        
        '''
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.y_transformer = y_transformer
        self.memory = memory
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.ordered_regressors = OrderedDict(self.regressors.items() if hasattr(self.regressors, 'items') else self.regressors)
    
    def _get_cv_predictions(self, estimator):
        if self.memory is None:
            return estimator.cv_predictions_
        else:
            return estimator.estimator_.cv_predictions_
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        
        # Create internal cross-validating estimators
        self.cross_validating_estimators_ = [(k, CrossValidatingEstimator(v, cv=self.cv, n_jobs=self.n_jobs, 
                                     verbose=self.verbose, 
                                     pre_dispatch=self.pre_dispatch)) for k, v in self.ordered_regressors.items()]
        if self.memory is not None:
            self.cross_validating_estimators_ = [(k, memorize(self.memory, v)) for k, v in self.cross_validating_estimators_]
        self.cross_validating_estimators_ = OrderedDict(self.cross_validating_estimators_)
        
        # Fit the inner regressors using cross-validation
        for est_name, est in self.cross_validating_estimators_.items():
            if self.verbose > 0:
                print('Super learner is fitting %s...' % est_name)
            est.fit(**fit_args)
            if self.verbose > 0:
                print('Super learner finished fitting %s.' % est_name)
        
        # Fit y_transformer if necessary and construct the fitting arguments for the 
        # meta-regressor.
        meta_fit_args = assoc(fit_args, 'X', 
                              np.concatenate(tuple(map(growd(2), 
                                                       [self._get_cv_predictions(est) for est in 
                                                        self.cross_validating_estimators_.values()])), axis=1))
        if self.y_transformer is not None:
            self.y_transformer_ = clone(self.y_transformer).fit(**fit_args)
            meta_fit_args = assoc(meta_fit_args, 'y',
                                  self.y_transformer_.transform(**dissoc(fit_args, 'sample_weight', 'y')))
        else:
            self.y_transformer_ = None
        
        # Fit the outer meta-regressor.  Cross validation is not used here.  Instead,
        # users of the SuperLearner are free to wrap the SuperLearner in a 
        # CrossValidatingEstimator.
        if self.verbose > 0:
            print('Super learner fitting meta-regressor...')
        self.meta_regressor_ = clone(self.meta_regressor).fit(**meta_fit_args)
        if self.verbose > 0:
            print('Super learner meta-regressor fitting complete.')
            
        # All scikit-learn compatible estimators must return self from fit
        return self
    
    def transform(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        return np.concatenate(tuple(map(growd(2), [est.predict(**args) for est in self.cross_validating_estimators_.values()])), axis=1)
    
    def predict(self, X, exposure=None):
        return self.meta_regressor_.predict(self.transform(X, exposure=exposure))

@syms.register(SuperLearner)
def syms_super_learner(estimator):
    return syms(first(estimator.cross_validating_estimators_.values()))

@sym_transform.register(SuperLearner)
def sym_transform_super_learner(estimator):
#     inputs = syms(estimator)
#     Var = VariableFactory(existing=inputs)
    return cart(*(map(sym_predict, estimator.cross_validating_estimators_.values())))
#     for est in estimator.cross_validating_estimators_:
#         part = sym_predict(est)
# #         outps = tuple(Var() for _ in part.outputs)
#         calls.append((outps, (part, inputs)))
#     calls = tuple(calls)
#     return Function(inputs, tuple(), outputs)

@sym_predict.register(SuperLearner)
def sym_predict_super_learner(estimator):
    return sym_predict(estimator.meta_regressor_).compose(sym_transform(estimator))


