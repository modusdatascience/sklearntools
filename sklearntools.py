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
from sklearn.utils.validation import check_X_y
import warnings
from sklearn.utils import safe_sqr
# import inspect
# 
# class CalibratingEstimator(BaseEstimator):
#     def __init__(self, estimator, calibrator):
#         pass
# 
# class BaseLearnNode(object):
#     def __init__(self, estimator, parents, children, pass_through):
#         self.estimator = estimator
#         self.parents = parents
#         self.children = children
#         self.pass_through = pass_through
#         
#     def _init_estimator_methods(self):
#         self.fit_arg_names, self.fit_var_args, 
#         self.fit_keywords, self.fit_defaults = inspect.getargspec(self.estimator.fit)
#         self.predict_arg_names, self.predict_var_args, 
#         self.predict_keywords, self.predict_defaults = inspect.getargspec(self.estimator.predict)
#         self.transform_arg_names, self.transform_var_args, 
#         self.transform_keywords, self.transform_defaults = inspect.getargspec(self.estimator.transform)
#     
#     def fit(self, X=None, y=None, sample_weight=None, **kwargs):
#         pass
#         
#         
# class LearnNode(BaseLearnNode):
#     def __init__(self, estimator, parents, children, pass_through):
#         '''
#         parents : dict {parent_estimator: [(name, ), ...]}
#         children : dict {method: child_estimators}
#         '''
#         super(self, LearnNode).__init__(estimator, parents, children, pass_through)
#         
# class InputNode(BaseLearnNode):
#     def __init__(self, estimator, children, pass_through):
#         '''
#         children : dict {method: child_estimators}
#         '''
#         super(self, InputNode).__init__(estimator, None, children, pass_through)
#         
# class OutputNode(BaseLearnNode):
#     def __init__(self, estimator, parents, pass_through):
#         '''
#         parents : dict {parent_estimator: (names...)}
#         '''
#         super(self, OutputNode).__init__(estimator, parents, None)
#         
# class LearnGraph(BaseEstimator):
#     def __init__(self, output_node):
#         self.output_node = output_node
#         
# 
# def pack(X, y, sample_weight):
#     return (X, y, sample_weight)
# #     if len(X.shape) > 1:
# #         n_x = X.shape[1]
# #     else:
# #         n_x = 1
# #     if y is not None and len(y.shape) > 1:
# #         n_y = y.shape[1]
# #     elif y is not None:
# #         n_y = 1
# #     else:
# #         n_y = 0
# #     if sample_weight is not None and len(sample_weight.shape) > 1:
# #         n_w = sample_weight.shape[1]
# #     elif sample_weight is not None:
# #         n_w = 1
# #     else:
# #         n_w = 0
# #     
# #     if X.shape[0] > 3:
# #         result = np.empty(shape=X.shape[0], n_x + n_y + n_w + 1)
# #         result[0,-1] = n_x
# #         result[1,-1] = n_y
# #         result[2,-1] = n_w
# #         result[:, :n_x] = X
# #         result[:, n_x:(n_x+n_y)] = y
# #         result[:, (n_x+n_y):(n_x+n_y+n_w)] = sample_weight
# #         
# #     else:
# #         result = np.empty(shape=X.shape[0], n_x + n_y + n_w + 3)
# #         result[0,-1] = n_x
# #         result[0,-2] = n_y
# #         result[0,-3] = n_w
# #         result[:, :n_x] = X
# #         result[:, n_x:(n_x+n_y)] = y
# #         result[:, (n_x+n_y):(n_x+n_y+n_w)] = sample_weight
# #     return result

#     
# 
# class Packer(BaseEstimator):
#     '''
#     A bit of a hack to get more functionality out of Pipeline.
#     '''
#     
#     def fit(self, X, y=None, sample_weight=None, *args, **kwargs):
#         return self
#     
#     def transform(self, X, y=None, sample_weight=None):
#         return pack(X, y, sample_weight)
#         
# class PackedEstimator(BaseEstimator):
#     '''
#     A bit of a hack to get more functionality out of Pipeline.
#     '''
#     def __init__(self, estimator):
#         self.estimator = estimator
#         
#     def fit(self, X, y=None, *args, **kwargs):
#         X, y, sample_weight = X
#         self.estimator_ = clone(self.estimator).fit(X, y, *args, **kwargs)
# #         self.
# 
# class Unpacker(BaseEstimator):
#     '''
#     A bit of a hack to get more functionality out of Pipeline.
#     '''
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

class FeatureImportanceEstimatorCV(STSimpleEstimator):
    def __init__(self, estimator, cv=None, scoring=None, score_combiner=None, 
                 n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
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
        feature_importances = []
        
        # For each feature, remove that feature and get the cross-validation scores
        for col in range(n_features):
            test_features = np.ones(shape=n_features, dtype=bool)
            
            if n_features > 1:
                test_features[col] = False
            
            scores = parallel(delayed(_fit_and_score)(clone(self.estimator), X[:, test_features], y, scorer,
                                      train, test, self.verbose, None,
                                      kwargs)
                              for train, test in cv)
            score = combiner(scores)
            feature_importances.append(score)
        self.feature_importances_ = np.array(feature_importances)
        
        # Finally, fit on the full data set with the selected set of features
        self.estimator_ = clone(self.estimator).fit(X, y, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X, *args, **kwargs)

class STSelector(STSimpleEstimator, SelectorMixin, MetaEstimatorMixin, TransformerMixin):
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

# class RFE(STSelector):
#     def __init__(self, estimator, n_features_to_select=None, step=1,
#                  estimator_params=None, verbose=0):
#         self.estimator = estimator
#         self.n_features_to_select = n_features_to_select
#         self.step = step
#         self.estimator_params = estimator_params
#         self.verbose = verbose
#         
#     def fit(self, X, y):
#         """Fit the RFE model and then the underlying estimator on the selected
#            features.
# 
#         Parameters
#         ----------
#         X : {array-like, sparse matrix}, shape = [n_samples, n_features]
#             The training input samples.
# 
#         y : array-like, shape = [n_samples]
#             The target values.
#         """
#         return self._fit(X, y)
#     
#     def _fit(self, X, y, step_score=None):
#         X, y = check_X_y(X, y, "csc", multi_output=True)
#         # Initialization
#         n_features = X.shape[1]
#         if self.n_features_to_select is None:
#             n_features_to_select = n_features // 2
#         else:
#             n_features_to_select = self.n_features_to_select
# 
#         if 0.0 < self.step < 1.0:
#             step = int(max(1, self.step * n_features))
#         else:
#             step = int(self.step)
#         if step <= 0:
#             raise ValueError("Step must be >0")
# 
#         if self.estimator_params is not None:
#             warnings.warn("The parameter 'estimator_params' is deprecated as "
#                           "of version 0.16 and will be removed in 0.18. The "
#                           "parameter is no longer necessary because the value "
#                           "is set via the estimator initialisation or "
#                           "set_params method.", DeprecationWarning)
# 
#         support_ = np.ones(n_features, dtype=np.bool)
#         ranking_ = np.ones(n_features, dtype=np.int)
# 
#         if step_score:
#             self.scores_ = []
# 
#         # Elimination
#         while np.sum(support_) > n_features_to_select:
#             # Remaining features
#             features = np.arange(n_features)[support_]
# 
#             # Rank the remaining features
#             estimator = clone(self.estimator)
#             if self.estimator_params:
#                 estimator.set_params(**self.estimator_params)
#             if self.verbose > 0:
#                 print("Fitting estimator with %d features." % np.sum(support_))
# 
#             estimator.fit(X[:, features], y)
# 
#             # Get coefs
#             if hasattr(estimator, 'coef_'):
#                 coefs = estimator.coef_
#             elif hasattr(estimator, 'feature_importances_'):
#                 coefs = estimator.feature_importances_
#             else:
#                 raise RuntimeError('The classifier does not expose '
#                                    '"coef_" or "feature_importances_" '
#                                    'attributes')
# 
#             # Get ranks
#             if coefs.ndim > 1:
#                 ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
#             else:
#                 ranks = np.argsort(safe_sqr(coefs))
# 
#             # for sparse case ranks is matrix
#             ranks = np.ravel(ranks)
# 
#             # Eliminate the worse features
#             threshold = min(step, np.sum(support_) - n_features_to_select)
# 
#             # Compute step score on the previous selection iteration
#             # because 'estimator' must use features
#             # that have not been eliminated yet
#             if step_score:
#                 self.scores_.append(step_score(estimator, features))
#             try:
#                 support_[features[ranks][:threshold]] = False
#             except:
#                 support_[features[ranks][:threshold]] = False
#             ranking_[np.logical_not(support_)] += 1
# 
#         # Set final attributes
#         features = np.arange(n_features)[support_]
#         self.estimator_ = clone(self.estimator)
#         if self.estimator_params:
#             self.estimator_.set_params(**self.estimator_params)
#         self.estimator_.fit(X[:, features], y)
# 
#         # Compute step score when only n_features_to_select features left
#         if step_score:
#             self.scores_.append(step_score(self.estimator_, features))
#         self.n_features_ = support_.sum()
#         self.support_ = support_
#         self.ranking_ = ranking_
# 
#         return self
# 
# class BRFE(STSelector):
#     '''
#     Performs recursive feature elimination, but chooses the best set of features
#     instead of just pruning to a pre-selected number.
#     '''
#     def __init__(self, estimator, min_features_to_select=1, step=1,
#                  verbose=0):
#         self.estimator = estimator
#         self.min_features_to_select = min_features_to_select
#         self.step = step
#         self.verbose = verbose
#     
#     def fit(self, X, y, *args, **kwargs):
#         # Fit the RFE, which is most of the work
#         rfe = RFE(clone(self.estimator), self.min_features_to_select,
#                               self.step, self.verbose)
#         rfe.fit(X, y, *args, **kwargs)
#         
#         # Find the best scoring step from the RFE
#         best_score = float('-inf')
#         best_idx = None
#         for i, score in enumerate(rfe.scores_):
#             if score >= best_score:
#                 best_score = score
#                 best_idx = i
#                 
#         # Calculate the mask for that step
#         thresh = len(rfe.scores_ - best_idx)
#         mask = rfe.ranking_ <= thresh
#         
#         # Set fitted attributes
#         self.support_ = mask
#         self.n_input_features_ = np.sum(self.support_)
#         self.rfe_ = rfe
#         self.self.estimator_ = clone(self.estimator).fit(X[:, self.support_], y, **kwargs)
#         
#     def predict(self, X, **kwargs):
#         if X.shape[1] == self.n_input_features_:
#             return self.estimator_.predict(X[:, self.support_], **kwargs)
#         elif X.shape[1] == self.n_features_:
#             return self.estimator_.predict(X, **kwargs)
#         else:
#             raise IndexError('X does not have the right number of columns')
#         

class BackwardEliminationEstimatorCV(STSimpleEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, cv=None, scoring=None, score_combiner=None, 
                 n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.score_combiner = score_combiner
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
    
    def _get_support_mask(self):
        return self.support_
    
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
        
        # Do stepwise backward elimination to find best feature set
        step_features = np.ones(shape=n_features, dtype=bool)
        best_score = -float('inf')
        best_support = None
        best_n_features = None
        elimination = []
        elimination_scores = []
        while np.sum(step_features) > 1:
            best_step_score = -float('inf')
            best_step_support = None
            best_step_col = None
            for col in range(n_features):
                if step_features[col]:
                    test_features = step_features.copy()
                    test_features[col] = False
                    
                    scores = parallel(delayed(_fit_and_score)(clone(self.estimator), X[:, test_features], y, scorer,
                                              train, test, self.verbose, None,
                                              kwargs)
                                      for train, test in cv)
                    score = combiner(scores)
                    if score > best_step_score:
                        best_step_score = score
                        best_step_support = test_features
                        best_step_col = col
                    
            elimination.append(best_step_col)
            elimination_scores.append(best_step_score)
            step_features[best_step_col] = False
            if best_step_score > best_score:
                best_score = best_step_score
                best_support = best_step_support
                best_n_features = np.sum(step_features)
                
        # Set attributes for best feature set
        self.n_input_features_ = n_features
        self.n_features_ = best_n_features
        self.support_ = best_support
        self.elimination_sequence_ = np.array(elimination)
        self.elimination_score_sequence_ = np.array(elimination_scores)
        
        # Finally, fit on the full data set with the selected set of features
        self.estimator_ = clone(self.estimator).fit(X[:, self.support_], y, **kwargs)
        
        return self

#     def predict(self, X, **kwargs):
#         if X.shape[1] == self.n_input_features_:
#             return self.estimator_.predict(X[:, self.support_], **kwargs)
#         elif X.shape[1] == self.n_features_:
#             return self.estimator_.predict(X, **kwargs)
#         else:
#             raise IndexError('X does not have the right number of columns')


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

