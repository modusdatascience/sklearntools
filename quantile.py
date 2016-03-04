from cvxpy import Variable, Minimize, Problem
from cvxpy.settings import OPTIMAL
import numpy as np
from sklearntools import STSimpleEstimator

class QuantileRegressor(STSimpleEstimator):
    _estimator_type = 'regressor'
    def __init__(self, q, prevent_crossing=True, lower_bound=None, upper_bound=None):
        self.q = np.asarray(q)
        self.prevent_crossing = prevent_crossing
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = sample_weight
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
    
    def _deviance(self, y, y_pred, sample_weight):
        y_diff = y_pred - np.repeat(y, len(self.q), axis=1)
        s = y_diff * (y_diff > 0)
        t = -y_diff * (y_diff < 0)
        q = np.repeat(np.reshape(self.q, (self.q.shape[0], 1)), y.shape[0], axis=1).T
        return np.sum(s * sample_weight * q + t * sample_weight * (1. - q))
    
    def score(self, X, y, sample_weight=None):
        if len(y.shape) == 1:
            y = y[:, None]
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = sample_weight
        if len(w.shape) == 1:
            w = w[:, None]
        y_pred = self.predict(X)
        m = y.shape[0]
        n = y.shape[1]
        base_pred = np.percentile(y, 100. * self.q, axis=0)[:,0] * np.ones(shape=(m, n*len(self.q)))
        tot = self._deviance(y, y_pred, w)
        base_tot = self._deviance(y, base_pred, w)
        return 1. - (tot / base_tot)
    
        
        
