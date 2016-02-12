'''
Created on Feb 11, 2016

@author: jason
'''
from sklearn.base import BaseEstimator, clone
import numpy as np
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression

class VariableSubsetSelector(BaseEstimator):
    pass
class MultipleResponseRegressor(BaseEstimator):
    def __init__(self, base_regressors):
        self.base_regressors = base_regressors
    
    def fit(self, X, y, fit_args=None, *args, **kwargs):
        if fit_args is None:
            fit_args = {}
        self.regressors_ = {}
        for columns, model in self.base_regressors.iteritems():
            dargs = kwargs.copy()
            dargs.update(fit_args.get(columns, {}))
            self.regressors_[columns] = clone(model).fit(X, y[:, columns], *args, **dargs)
        return self
    
    def predict(self, X, predict_args=None, *args, **kwargs):
        if predict_args is None:
            predict_args = {}
        predictions = []
        for columns, model in self.regressors_.iteritems():
            dargs = kwargs.copy()
            dargs.update(predict_args.get(columns, {}))
            prediction = model.predict(X, *args, **dargs)
            predictions.append(prediction if len(prediction.shape) == 2 else prediction[:, None])
        return np.concatenate(predictions, axis=1)
    
    
class ProbaPredictingRegressor(BaseEstimator):
    def __init__(self, base_regressor):
        self.base_regressor = base_regressor
    
    def fit(self, X, y, *args, **kwargs):
        self.regressor_ = clone(self.base_regressor)
        self.regressor_.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        return self.regressor_.predict_proba(X, *args, **kwargs)
    
    
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
        
    model = MultipleResponseRegressor({0: LinearRegression(), 
                                       1:ProbaPredictingRegressor(LogisticRegression())})
    model.fit(X, y)
    
    assert np.mean(beta1 - model.regressors_[0].coef_) < .01
    assert np.mean(beta2 - model.regressors_[1].regressor_.coef_) < .01
    model.get_params()
    model.predict(X)


if __name__ == '__main__':
    

    test_multiple_response_regressor()
    print 'Success!'
    
    