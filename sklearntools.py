'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin, is_classifier,\
    is_regressor
import numpy as np
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import check_cv, cross_val_score, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib.parallel import Parallel, delayed

class QuantileRegressor(BaseEstimator):
    _estimator_type = 'regressor'

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
    y = np.dot(X, beta) + 0.1 * np.random.normal(size=(m, 1))
    
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
    test_multiple_response_regressor()
    test_backward_elimination_estimator_cv()
    print 'Success!'
    
    