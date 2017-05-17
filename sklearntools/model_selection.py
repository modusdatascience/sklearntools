from sklearntools import STSimpleEstimator, BaseDelegatingEstimator,\
    _fit_and_score, standard_methods, non_fit_methods, safer_call
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


class ModelSelector(BaseDelegatingEstimator):
    def __init__(self, candidates, scoring=None):
        self._process_candidates(candidates)
        self.scoring = scoring
    
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
        # Extract arguments
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight,
                                      exposure=exposure)
        predict_args = fit_args.copy()
        if 'X' in predict_args:
            del predict_args['X']
        if 'sample_weight' in predict_args:
            del predict_args['sample_weight']
        
        # Find the best scoring candidate
        best_score = float('-inf')
        best_candidate = None
        self.candidates_ = []
        self.candidate_scores_ = []
        for i, candidate in enumerate(self.candidates):
            scorer = check_scoring(candidate, scoring=self.scoring)
            
            # Do the actual fitting and scoring
            candidate_ = clone(candidate).fit(**fit_args)
            if self.scoring is None and hasattr(candidate_, 'score_'):
                score = candidate_.score_
            else:
                score = safer_call(scorer, candidate_, **fit_args)
                    
#             score, _, candidate_ = _fit_and_score(clone(candidate), fit_args, scorer, slice(None), slice(None))
            
            # Store the results
            self.candidate_scores_.append(score)
            self.candidates_.append(candidate_)
            
            # If it's the best so far, keep it
            if score > best_score:
                best_score = score
                best_candidate = candidate_
                best_candidate_index = i
        
        self.best_estimator_ = best_candidate
        self.best_score_ = best_score
        self.best_candidate_index_ = best_candidate_index
        self.best_estimator = best_candidate
        self._create_delegates('best_estimator', non_fit_methods)
        del self.best_estimator
        # Fit the best candidate
        
#         self.best_estimator = best_candidate
#         self.best_estimator_ = clone(best_candidate).fit(**fit_args)
#         self.best_score_ = best_score
#         self._create_delegates('best_estimator', non_fit_methods)
#         del self.best_estimator
        return self
            
            
            
            
            
        