'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin, is_classifier,\
    is_regressor, TransformerMixin
import numpy as np
from sklearn.cross_validation import check_cv, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib.parallel import Parallel, delayed
from cvxpy import Variable, Minimize, Problem
from cvxpy.settings import OPTIMAL
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import safe_mask
    
class SklearnTool(object):
    pass

class STEstimator(BaseEstimator, SklearnTool):
    def _get_steps(self, other):
        if isinstance(other, Pipeline):
            steps = [step for step in other.steps]
            
        else:
            other_name = other.__class__.__name__
            steps = [(other_name, other)]
        return steps
    
    def _get_name(self, steps):
        step_names = set([step[0] for step in steps])
        self_class_name = self.__class__.__name__
        self_name = self_class_name
        i = 2
        while True:
            if self_name in step_names:
                self_name = self_class_name + '_' + str(i)
            else:
                break
            i += 1
            if i > 1e6:
                raise ValueError('Unable to name estimator %s in pipeline' % str(self))
        return self_name, steps

class STSimpleEstimator(STEstimator):
    def __or__(self, other):
        '''
        self | other
        '''
        steps = self._get_steps(other)
        self_name = self._get_name(steps)
        steps = [(self_name, self)] + steps
        return STPipeline(steps)
        
    def __ror__(self, other):
        '''
        other | self
        '''
        steps = self._get_steps(other)
        self_name = self._get_name(steps)
        steps = steps + [(self_name, self)]
        return STPipeline(steps)

class STPipeline(STEstimator, Pipeline):
    def __or__(self, other):
        '''
        self | other
        '''
        other_steps = self._get_steps(other)
        steps = self.steps + other_steps
        return STPipeline(steps)
        
    def __ror__(self, other):
        '''
        other | self
        '''
        other_steps = self._get_steps(other)
        steps = other_steps + self.steps
        return STPipeline(steps)

class IdentityTransformer(STSimpleEstimator, TransformerMixin):
    def __init__(self):
        pass
     
    def fit(self, X, y=None, sample_weight=None):
        pass
     
    def transform(self, X, y=None):
        return X

class ResponseTransformingEstimator(STSimpleEstimator, TransformerMixin):
    def __init__(self, estimator, transformer, inverter=IdentityTransformer()):
        self.estimator = estimator
        self.transformer = transformer
        self.inverter = inverter
        
    def fit(self, X, y, transformer_args, estimator_args, inverter_args):
        self.transformer_ = clone(self.transformer).fit(y, **transformer_args)
        y_transformed = self.transformer_.transform(y)
        self.estimator_ = clone(self.estimator).fit(X, y_transformed, **estimator_args)
        y_predicted = self.estimator_.predict(X)
        self.inverter_ = clone(self.inverter).fit(y_predicted, y)
        return self
    
    def predict(self, X, transformer_args, estimator_args, inverter_args):
        return self.inverter_.transform(self.estimator_.predict(X))
        
# class LogTransformer(BaseEstimator, TransformerMixin):
#     _estimator_type = 'transformer'
#     def __init__(self, offset=None, fixed_offset=1.):
#         self.offset = offset
#         self.fixed_offset = fixed_offset
#     
#     def fit(self, X, y=None):
#         pass
#     
#     def transform(self, X, y=None):
#         pass
# 
# class BoxCoxTransformer(BaseEstimator, TransformerMixin):
#     _estimator_type = 'transformer'
#     
#     def fit(self, X, y=None):
#         pass
#     
#     def transform(self, X, y=None):
#         pass
#     

class QuantileRegressor(STSimpleEstimator):
    _estimator_type = 'regressor'
    def __init__(self, q, prevent_crossing=True, lower_bound=None, upper_bound=None):
        self.q = q
        self.prevent_crossing = prevent_crossing
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def fit(self, X, y, w):
        m, n = X.shape
        p = len(self.q)
#         variables = []
#         s_vars = []#Variable(m)
#         t_vars = []#Variable(m)
#         b_vars = []#Variable(n)
        constraints = []
        objective_function = 0.0
        last_s = None
        last_t = None
        b_vars = []
        for i in range(p):
            s = Variable(m)
            t = Variable(m)
            b = Variable(n)
            b_vars.append(b)
            tau = self.q[i]
            constraints.append(s >= 0)
            constraints.append(t >= 0)
            constraints.append(y - X * b == s - t)
            objective_function += tau * w * s + \
                                  (1 - tau) * w * t
            if i >= 1 and self.prevent_crossing:
                constraints.append(s <= last_s)
                constraints.append(t >= last_t)
            if self.lower_bound is not None:
                constraints.append(X * b >= self.lower_bound)
            if self.upper_bound is not None:
                constraints.append(X * b <= self.upper_bound)
            last_s = s
            last_t = t
        objective = Minimize(objective_function)
        problem = Problem(objective, constraints)
        problem.solve()
        if problem.status != OPTIMAL:
            raise ValueError('Problem status is: %s' % problem.status)
        self.coef_ = np.array([np.ravel(b.value) for b in b_vars]).T
        return self
    
    def predict(self, X):
        return np.dot(X, self.coef_)

def weighted_average_score_combine(scores):
    scores_arr = np.array(scores)
    return np.average(scores_arr[0,:], weights=scores_arr[:,1])
    
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
    
class MultipleResponseEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, base_estimators):
        self.base_estimators = base_estimators
        
    @property
    def _estimator_type(self):
        if all([is_classifier(estimator) for estimator in self.base_estimators.values()]):
            return 'classifier'
        elif all([is_regressor(estimator) for estimator in self.base_estimators.values()]):
            return 'regressor'
        else:
            return 'mixed'
    
    def fit(self, X, y, fit_args=None, *args, **kwargs):
        if fit_args is None:
            fit_args = {}
        self.estimators_ = []
        for name, columns, model in self.base_estimators:
            dargs = kwargs.copy()
            dargs.update(fit_args.get(name, {}))
            
            # Select the appropriate columns
            y_ = y[:, columns]
            if y_.shape[1] == 1:
                y_ = y_[:, 0]
            
            # Fit the estimator
            self.estimators_.append((name, columns, clone(model).fit(X, y_, *args, **dargs)))
        self.estimators_dict_ = {name: (columns, model) for name, columns, model in self.estimators_}
        
        # Do a prediction on a single row of data for each estimator in order to 
        # determine the number of predicted columns for each one
        X_ = X[0:1, :]
        self.prediction_columns_ = []
        for name, columns, model in self.estimators_:
            prediction = model.predict(X_)
            if len(prediction.shape) > 1:
                n_columns = prediction.shape[1]
            else:
                n_columns = 1
            self.prediction_columns_ += [name] * n_columns
        
        return self
    
    def predict(self, X, predict_args=None, *args, **kwargs):
        if predict_args is None:
            predict_args = {}
        predictions = []
        for name, columns, model in self.estimators_:  # @UnusedVariable
            dargs = kwargs.copy()
            dargs.update(predict_args.get(name, {}))
            prediction = model.predict(X, *args, **dargs)
            predictions.append(prediction if len(prediction.shape) == 2 else prediction[:, None])
        return np.concatenate(predictions, axis=1)
    
    
class ProbaPredictingEstimator(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    
    def fit(self, X, y, *args, **kwargs):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict_proba(X, *args, **kwargs)

