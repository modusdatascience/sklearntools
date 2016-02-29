from cvxpy import Variable, Minimize, Problem
from cvxpy.settings import OPTIMAL
import numpy as np
from sklearntools import STSimpleEstimator

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
