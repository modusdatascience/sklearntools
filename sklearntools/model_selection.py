from sklearntools import STSimpleEstimator, BaseDelegatingEstimator,\
    _fit_and_score, standard_methods, non_fit_methods
from sklearn.base import MetaEstimatorMixin, is_classifier, clone
from itertools import product
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib.parallel import Parallel, delayed
import numpy as np
from sklearn.metrics.scorer import check_scoring
from feature_selection import check_score_combiner
from calibration import no_cv

def candidate_grid(estimator, param_grid):
    pass

class ModelSelectorCV(BaseDelegatingEstimator):
    def __init__(self, candidates, scoring, score_combiner=None, cv=2, n_jobs=1, verbose=0, 
                 pre_dispatch='2*n_jobs'):
        self._process_candidates(candidates)
        self.scoring = scoring
        self.score_combiner = score_combiner
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.cv = cv
        self.verbose = verbose
    
    def _process_candidates(self, candidates):
        '''
        Convert candidates from whatever it is and assign self.candidates and self.candidate_names.
        '''
        converted = False
        try:
            pairs = list(candidates.items())
            converted = True
        except AttributeError:
            pass
        if not converted:
            try:
                pairs = [(cand[0], cand[1]) for cand in candidates]
            except IndexError:
                pass
            except TypeError:
                pass
        if not converted:
            pairs = [('candidate_%d' % i, candidate) for i, candidate in enumerate(candidates)]
        self.candidates = [pair[1] for pair in pairs]
        self.candidate_names = [pair[0] for pair in pairs]
    
    def iter_candidates(self):
        return list(self.candidates)
            
    def iter_candidate_names(self):
        return list(self.candidate_names)
    
    def iter_candidate_pairs(self):
        return zip(self.candidate_names, self.candidates)
    
    def iter_fitted_candidates(self):
        return list(self.candidates_)
    
    def iter_fitted_candidate_pairs(self):
        return zip(self.candidate_names, self.candidates_)
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        # For later
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        pre_dispatch=self.pre_dispatch)
        
        # Extract arguments
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        
        # Find the best scoring candidate
        best_score = float('-inf')
        best_candidate = None
        for candidate in self.candidates:
            # Sort out the parameters for this candidate
            if self.cv == 1:
                cv = no_cv(X=X, y=y)
            else:
                if hasattr(self.cv, 'split'):
                    cv = self.cv.split(X, y)
                else:
                    cv = check_cv(self.cv, X=X, y=y, classifier=is_classifier(candidate))
                    
            scorer = check_scoring(candidate, scoring=self.scoring)
            combiner = check_score_combiner(candidate, self.score_combiner)
            
            # Do the actual fitting and scoring
            scores = parallel(delayed(_fit_and_score)(clone(candidate), fit_args, scorer,
                                      train, test)
                              for train, test in cv)
            score = combiner(scores)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        # Fit the best candidate
        self.best_estimator = best_candidate
        self.best_estimator_ = clone(best_candidate).fit(**fit_args)
        self.best_score_ = best_score
        self._create_delegates('best_estimator', non_fit_methods)
        del self.best_estimator
        return self
            
            
            
            
            
        