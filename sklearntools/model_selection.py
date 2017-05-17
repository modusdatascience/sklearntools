from sklearntools import BaseDelegatingEstimator,\
    non_fit_methods, safer_call, sym_methods
from sklearn.base import clone
from sklearn.metrics.scorer import check_scoring

def candidate_grid(estimator, param_grid):
    pass

class ModelSelector(BaseDelegatingEstimator):
    def __init__(self, candidates, scoring=None, verbose=False):
        self.candidates = candidates
        self._process_candidates(candidates)
        self.scoring = scoring
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
        self.candidate_models = [pair[1] for pair in pairs]
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
    
    def _run_fit(self, method, X, y=None, sample_weight=None, exposure=None):
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
        outputs = []
        for i, candidate in enumerate(self.candidate_models):
            candidate_name = self.candidate_names[i]
            if self.verbose:
                print('Fitting candidate: %s' % candidate_name)
            scorer = check_scoring(candidate, scoring=self.scoring)
            
            # Do the actual fitting and scoring
            candidate_ = clone(candidate)
            output = getattr(candidate_, method)(**fit_args)
            outputs.append(output)
            if self.scoring is None and hasattr(candidate_, 'score_'):
                score = candidate_.score_
            else:
                score = safer_call(scorer, candidate_, **fit_args)
                    
            # Store the results
            self.candidate_scores_.append(score)
            self.candidates_.append(candidate_)
            
            # If it's the best so far, keep it
            if score > best_score:
                best_score = score
                best_candidate = candidate_
                best_candidate_index = i
                best_output = output
                best_candidate_name = candidate_name
        
        self.best_estimator_ = best_candidate
        self.best_score_ = best_score
        self.best_candidate_index_ = best_candidate_index
        self.best_candidate_name = best_candidate_name
        if self.verbose:
            print('Selected best candidate: %s' % self.best_candidate_name)
        self.best_estimator = best_candidate
        self._create_delegates('best_estimator', non_fit_methods + sym_methods)
        del self.best_estimator
        return best_output
        
    def fit(self, X, y=None, sample_weight=None, exposure=None):
        return self._run_fit('fit', X=X, y=y, sample_weight=sample_weight, exposure=exposure)
    
    def fit_predict(self, X, y=None, sample_weight=None, exposure=None):
        return self._run_fit('fit_predict', X=X, y=y, sample_weight=sample_weight, exposure=exposure)
            
        