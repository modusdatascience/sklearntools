import numpy as np
from sklearn.base import MetaEstimatorMixin, is_classifier, clone,\
    TransformerMixin
from sklearntools import STSimpleEstimator
from sklearn.cross_validation import check_cv, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import safe_mask

def weighted_average_score_combine(scores):
    scores_arr = np.array(scores)
    return np.average(scores_arr[:,0], weights=scores_arr[:,1])
    
def check_score_combiner(estimator, score_combiner):
    if score_combiner is None:
        return weighted_average_score_combine
    else:
        raise NotImplementedError('Score combiner %s not implemented' % str(score_combiner))

class SingleEliminationFeatureImportanceEstimatorCV(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, cv=None, scoring=None, check_constant_model=True, 
                 score_combiner=None, n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.check_constant_model = check_constant_model
        self.score_combiner = score_combiner
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    
    def fit(self, X, y, **kwargs):
        cv = check_cv(self.cv, X=X, y=y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        combiner = check_score_combiner(self.estimator, self.score_combiner)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        pre_dispatch=self.pre_dispatch)
        n_features = X.shape[1]
        feature_deletion_scores = []
        
        # Get cross-validated scores with all features present
        full_scores = parallel(delayed(_fit_and_score)(clone(self.estimator), X, y, scorer,
                                      train, test, self.verbose, None,
                                      kwargs)
                              for train, test in cv)
        self.score_ = combiner(full_scores)
        
        # For each feature, remove that feature and get the cross-validation scores
        for col in range(n_features):
            test_features = np.ones(shape=n_features, dtype=bool)
            
            if n_features > 1:
                test_features[col] = False
                
                scores = parallel(delayed(_fit_and_score)(clone(self.estimator), X[:, test_features], y, scorer,
                                          train, test, self.verbose, None,
                                          kwargs)
                                  for train, test in cv)
            elif self.check_constant_model:
                # If there's only one feature to begin with, do the fitting and scoring on a 
                # constant predictor.
                scores = parallel(delayed(_fit_and_score)(clone(self.estimator), np.ones(shape=(X.shape[0], 1)), 
                                          y, scorer, train, test, self.verbose, None, kwargs)
                                  for train, test in cv)
            else:
                scores = full_scores
            score = combiner(scores)
            feature_deletion_scores.append(score)
        
        # Higher scores are better.  Higher feature importance means the feature is more important. 
        # This code reconciles these facts.
        self.feature_importances_ = self.score_ - np.array(feature_deletion_scores)
        
        # Finally, fit on the full data set
        self.estimator_ = clone(self.estimator).fit(X, y, **kwargs)
    
        # A fit method should always return self for chaining purposes
        return self
    
    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)
    
    def score(self, X, y, sample_weight=None):
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        return scorer(self, X, y, sample_weight)

class STSelector(STSimpleEstimator, SelectorMixin, MetaEstimatorMixin, TransformerMixin):
    
    # Override transform method from SelectorMixin because it doesn't handle the
    # case of selecting zero features the way I want it to.
    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        mask = self.get_support()
        if not mask.any():
            return np.ones(shape=(X.shape[0], 1))
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return X[:, safe_mask(X, mask)]
    
    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)
    
    def _get_support_mask(self):
        return self.support_
    
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))

class BackwardEliminationEstimator(STSelector, MetaEstimatorMixin):
    def __init__(self, estimator, scoring=None, check_constant_model=True):
        self.estimator = estimator
        self.scoring = scoring
        self.check_constant_model = check_constant_model
    
    def fit(self, X, y, **kwargs):
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        sample_weight = kwargs.get('sample_weight', None)
        
        # Do stepwise backward elimination to find best feature set
        support = np.ones(shape=n_features, dtype=bool)
        best_score = -float('inf')
        best_support = None
        best_n_features = None
        elimination = []
        scores = []
        while np.sum(support) >= 1:
            
            # Fit the estimator 
            estimator = clone(self.estimator).fit(X[:, support], y, **kwargs)
            
            # Score the estimator
            if self.scoring is None and hasattr(estimator, 'score_'):
                score = estimator.score_
            else:
                score = scorer(estimator, X, y, sample_weight)
            scores.append(score)
            
            # Compare to previous tries
            if score > best_score:
                best_score = score
                best_support = support.copy()
                best_n_features = np.sum(support)
            
            # Remove the least important feature from the support for next time
            worst_feature = np.argmin(estimator.feature_importances_)
            worst_feature_idx = np.argwhere(support)[worst_feature][0]
            support[worst_feature_idx] = False
            elimination.append(worst_feature_idx)
            
        # Score a constant input model in case it's the best choice.
        # (This would mean the predictors are essentially useless.)
        if self.check_constant_model:
            # Fit the estimator 
            estimator = clone(self.estimator).fit(np.ones(shape=(X.shape[0],1)), y, **kwargs)
            
            # Score the estimator
            if self.scoring is None and hasattr(estimator, 'score_'):
                score = estimator.score_
            else:
                score = scorer(estimator, X, y, sample_weight)
            
            # Compare to previous tries
            if score > best_score:
                best_score = score
                best_support = np.zeros_like(support)
                best_n_features = 0
            scores.append(score)
        
        # Set attributes for best feature set
        self.n_input_features_ = n_features
        self.n_features_ = best_n_features
        self.support_ = best_support
        self.elimination_sequence_ = np.array(elimination)
        self.scores_ = np.array(scores)
        
        # Finally, fit on the full data set with the selected set of features
        self.estimator_ = clone(self.estimator).fit(X[:, self.support_], y, **kwargs)
        
        return self
    
