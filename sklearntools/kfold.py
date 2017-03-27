from sklearn.externals.joblib.parallel import Parallel, delayed
from calibration import no_cv
from sklearn.cross_validation import check_cv
from sklearn.base import is_classifier, clone
from sklearntools import _fit_and_predict, standard_methods, BaseDelegatingEstimator, safe_assign_subset
import numpy as np
from sym import sym_predict, syms, sym_predict_parts, sym_transform_parts

class CrossValidatingEstimator(BaseDelegatingEstimator):
    def __init__(self, estimator, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self._create_delegates('estimator', standard_methods)
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
    def sym_predict(self):
        return sym_predict(self.estimator_)
    
    def sym_predict_parts(self, target=None):
        return sym_predict_parts(self.estimator_, target)
    
    def sym_transform_parts(self, target=None):
        return sym_transform_parts(self.estimator_, target)
    
    def syms(self):
        return syms(self.estimator_)
    
    def fit_predict(self, X, y=None, sample_weight=None, exposure=None):
        # For later
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        pre_dispatch=self.pre_dispatch)
        
        # Extract arguments
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        
        # Sort out cv parameters
        if self.cv == 1:
            cv = no_cv(X=X, y=y)
        else:
            if hasattr(self.cv, 'split'):
                cv = self.cv.split(X, y)
            else:
                cv = check_cv(self.cv, X=X, y=y, classifier=is_classifier(self.estimator))
                
        # Do the cross validation fits
        cv_fits = parallel(delayed(_fit_and_predict)(clone(self.estimator), fit_args, train, test) for train, test in cv)
        
        # Combine predictions from cv fits
        prediction = np.empty_like(y)
        for fit in cv_fits:
            safe_assign_subset(prediction, fit[2], fit[1])
        
        # Store cross validation models
        self.cv_estimators_ = [fit[0] for fit in cv_fits]
        
        # Fit on entire data set
        self.fit(**fit_args)
        
        return prediction
