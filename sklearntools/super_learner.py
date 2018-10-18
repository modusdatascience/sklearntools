from sklearntools.sklearntools import STSimpleEstimator, growd
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

class SuperLearner(STSimpleEstimator):
    def __init__(self, regressors, meta_regressor, y_transformer=None, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        '''
        regressors : should be a dict-like structure or items-like
        '''
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.y_transformer = y_transformer
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.ordered_regressors = OrderedDict(self.regressors.items() if hasattr(self.regressors, 'items') else self.regressors)
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        
        # Create internal cross-validating estimators
        self.cross_validating_estimators_ = OrderedDict((k, CrossValidatingEstimator(v, cv=self.cv, n_jobs=self.n_jobs, 
                                     verbose=self.verbose, 
                                     pre_dispatch=self.pre_dispatch)) for k, v in self.ordered_regressors.items())
        
#         frozendict(valmap(lambda x:
#             CrossValidatingEstimator(x, cv=self.cv, n_jobs=self.n_jobs, 
#                                      verbose=self.verbose, 
#                                      pre_dispatch=self.pre_dispatch), self.regressors).items())
        
        # Fit the inner regressors using cross-validation
        for est_name, est in self.cross_validating_estimators_.items():
            if self.verbose > 0:
                print('Super learner is fitting %s...' % est_name)
            est.fit(**fit_args)
            if self.verbose > 0:
                print('Super learner finished fitting %s.' % est_name)
        
        # Fit the outer meta-regressor.  Cross validation is not used here.  Instead,
        # users of the SuperLearner are free to wrap the SuperLearner in a 
        # CrossValidatingEstimator.
        meta_fit_args = assoc(fit_args, 'X', 
                              np.concatenate(tuple(map(growd(2), [est.cv_predictions_ for est in self.cross_validating_estimators_.values()])), axis=1))
        if self.y_transformer is not None:
            self.y_transformer_ = clone(self.y_transformer).fit(**fit_args)
            meta_fit_args = assoc(meta_fit_args, 'y',
                                  self.y_transformer_.transform(**dissoc(fit_args, 'sample_weight', 'y')))
        
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


