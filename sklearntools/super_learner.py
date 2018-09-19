from sklearntools.sklearntools import STSimpleEstimator, growd
from sklearn2code.utility import tupify
from sklearn.base import clone
from sklearn.externals.joblib.parallel import Parallel
from sklearntools.kfold import CrossValidatingEstimator
import numpy as np
from toolz.dicttoolz import assoc, get_in, keyfilter
from operator import __contains__
from toolz.functoolz import curry
from sklearn2code.sym.base import sym_predict, syms, sym_transform
from sklearn2code.sym.function import VariableFactory, Function, cart

class SuperLearner(STSimpleEstimator):
    def __init__(self, regressors, meta_regressor, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        
        # Create internal cross-validating estimators
        self.cross_validating_estimators_ = tuple(map(lambda x:
            CrossValidatingEstimator(x, cv=self.cv, n_jobs=self.n_jobs, 
                                     verbose=self.verbose, 
                                     pre_dispatch=self.pre_dispatch), self.regressors))
        
        # Fit the inner regressors using cross-validation
        for est in self.cross_validating_estimators_:
            est.fit(**fit_args)
        
        # Fit the outer meta-regressor.  Cross validation is not used here.  Instead,
        # users of the SuperLearner are free to wrap the SuperLearner in a 
        # CrossValidatingEstimator.
        meta_fit_args = assoc(fit_args, 'X', 
                              np.concatenate(tuple(map(growd(2), [est.cv_predictions_ for est in self.cross_validating_estimators_])), axis=1))
        self.meta_regressor_ = clone(self.meta_regressor).fit(**meta_fit_args)
        
        # All scikit-learn compatible estimators must return self from fit
        return self
    
    def transform(self, X, exposure=None):
        args = self._process_args(X=X, exposure=exposure)
        return np.concatenate(tuple(map(growd(2), [est.predict(**args) for est in self.cross_validating_estimators_])), axis=1)
    
    def predict(self, X, exposure=None):
        return self.meta_regressor_.predict(self.transform(X, exposure=exposure))

@syms.register(SuperLearner)
def syms_super_learner(estimator):
    return syms(estimator.cross_validating_estimators_[0])

@sym_transform.register(SuperLearner)
def sym_transform_super_learner(estimator):
#     inputs = syms(estimator)
#     Var = VariableFactory(existing=inputs)
    return cart(*(map(sym_predict, estimator.cross_validating_estimators_)))
#     for est in estimator.cross_validating_estimators_:
#         part = sym_predict(est)
# #         outps = tuple(Var() for _ in part.outputs)
#         calls.append((outps, (part, inputs)))
#     calls = tuple(calls)
#     return Function(inputs, tuple(), outputs)

@sym_predict.register(SuperLearner)
def sym_predict_super_learner(estimator):
    return sym_predict(estimator.meta_regressor_).compose(sym_transform(estimator))


