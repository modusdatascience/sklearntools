'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin, is_classifier,\
    is_regressor, TransformerMixin
import numpy as np
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import check_cv, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib.parallel import Parallel, delayed

from cvxpy import Variable, Minimize, Problem
from cvxpy.settings import OPTIMAL
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
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
     
    def fit(self, X, y=None, sample_weight=None):
        pass
     
    def transform(self, X, y=None):
        return X

class ResponseTransformingEstimator(BaseEstimator, TransformerMixin):
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
class VariableImportanceEstimatorCV(BaseEstimator):
    def __init__(self, estimator, cv=None, scoring=None, score_combiner=None, 
                 n_jobs=1, verbose=0, pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.score_combiner = score_combiner
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

class QuantileRegressor(BaseEstimator):
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

def sim_quantiles(taus, quantiles):
    assert quantiles.shape[1] == len(taus)
    p_vals = np.array([taus[i] - taus[i-1] for i in range(1,len(taus))])
    choices = np.random.multinomial(1, p_vals, size=quantiles.shape[0])
    lowers = quantiles[:, :-1][choices==1]
    uppers = quantiles[:, 1:][choices==1]
    y = np.random.uniform(low=lowers, high=uppers)
    return y

def test_quantile_regression():
    np.random.seed(1)
    X = np.random.uniform(0,1,size=(10000,2))
    b = np.array([[0,0],[1,1],[2,2],[4,4],[6,6]]).transpose()
    quantiles = np.dot(X, b)
    taus = np.array([0.,.1,.5,.9,1.])
    y = sim_quantiles(taus, quantiles)
    w = np.ones_like(y)
    qr = QuantileRegressor(taus[1:-1]).fit(X, y, w)
    assert np.max(np.abs(qr.coef_ - b[:, 1:-1]) < .2)
    pred = qr.predict(X)
    assert np.max(pred - quantiles[:,1:-1]) < .2
    
    # Check for crossing
    y_hat = qr.predict(X)
    for i in range(y_hat.shape[1] - 1):
        assert np.all(y_hat[:,i] <= y_hat[:,i+1])
    
    # Test lower bound
    qr = QuantileRegressor(taus[1:-1], lower_bound=1.).fit(X, y, w)
    assert np.min(qr.predict(X)) >= 0.9999999999
    
    # Test upper bound
    qr = QuantileRegressor(taus[1:-1], upper_bound=10.).fit(X, y, w)
    assert np.max(qr.predict(X)) <= 10.00000000001
    
    # Test both bounds
    qr = QuantileRegressor(taus[1:-1], lower_bound=0.5, upper_bound=75.).fit(X, y, w)
    assert np.min(qr.predict(X)) >= 0.4999999999
    assert np.max(qr.predict(X)) <= 75.0000000001
    
    # Unconstrained
    qr = QuantileRegressor(taus[1:-1], prevent_crossing=False).fit(X, y, w)
    
def weighted_average_score_combine(scores):
    scores_arr = np.array(scores)
    return np.average(scores_arr[0,:], weights=scores_arr[1,:])
    
def check_score_combiner(estimator, score_combiner):
    if score_combiner is None:
        return weighted_average_score_combine
    else:
        raise NotImplementedError('Score combiner %s not implemented' % str(score_combiner))

class BackwardEliminationEstimatorCV(BaseEstimator, MetaEstimatorMixin):
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

    def predict(self, X, **kwargs):
        if X.shape[1] == self.n_input_features_:
            return self.estimator_.predict(X[:, self.support_], **kwargs)
        elif X.shape[1] == self.n_features_:
            return self.estimator_.predict(X, **kwargs)
        else:
            raise IndexError('X does not have the right number of columns')

def test_backward_elimination_estimator_cv():
    np.random.seed(1)
    m = 100000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    y = np.dot(X, beta) + 0.01 * np.random.normal(size=(m, 1))
    
    target_sequence = np.ravel(np.argsort(beta ** 2, axis=0)[::-1])
    model = BackwardEliminationEstimatorCV(LinearRegression())
    model.fit(X, y)
    np.testing.assert_array_equal(model.elimination_sequence_, target_sequence[:-1])
    

class MultipleResponseEstimator(BaseEstimator, MetaEstimatorMixin):
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
    
    
class ProbaPredictingEstimator(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    
    def fit(self, X, y, *args, **kwargs):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict_proba(X, *args, **kwargs)
    
    
def test_multiple_response_regressor():
    np.random.seed(1)
    m = 100000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta1 = np.random.normal(size=(n,1))
    beta2 = np.random.normal(size=(n,1))
        
    y1 = np.dot(X, beta1)
    p2 = 1. / (1. + np.exp( - np.dot(X, beta2)))
    y2 = np.random.binomial(n=1, p=p2)
    y = np.concatenate([y1, y2], axis=1)
        
    model = MultipleResponseEstimator([('linear', np.array([True, False], dtype=bool), LinearRegression()), 
                                       ('logistic', np.array([False, True], dtype=bool), ProbaPredictingEstimator(LogisticRegression()))])
    model.fit(X, y)
    
    assert np.mean(beta1 - model.estimators_dict_['linear'][1].coef_) < .01
    assert np.mean(beta2 - model.estimators_dict_['logistic'][1].estimator_.coef_) < .01
    assert model.prediction_columns_ == ['linear', 'logistic', 'logistic']
    model.get_params()
    model.predict(X)


if __name__ == '__main__':
    test_quantile_regression()
    test_backward_elimination_estimator_cv()
    test_multiple_response_regressor()
    print 'Success!'
    
    