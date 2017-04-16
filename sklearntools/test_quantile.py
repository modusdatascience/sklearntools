import numpy as np
from functools import wraps
from nose import SkipTest

def if_cvxpy(func):
    """Test decorator that skips test if statsmodels not installed. """

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import cvxpy
        except ImportError:
            raise SkipTest('cvxpy not available.')
        else:
            return func(*args, **kwargs)
    return run_test

def sim_quantiles(taus, quantiles):
    assert quantiles.shape[1] == len(taus)
    p_vals = np.array([taus[i] - taus[i-1] for i in range(1,len(taus))])
    choices = np.random.multinomial(1, p_vals, size=quantiles.shape[0])
    lowers = quantiles[:, :-1][choices==1]
    uppers = quantiles[:, 1:][choices==1]
    y = np.random.uniform(low=lowers, high=uppers)
    return y

@if_cvxpy
def test_quantile_regression():
    from .quantile import QuantileRegressor
    np.random.seed(1)
    X = np.random.uniform(0,1,size=(10000,2))
    b = np.array([[0,0],[1,1],[2,2],[4,4],[6,6]]).transpose()
    quantiles = np.dot(X, b)
    taus = np.array([0.,.1,.5,.9,1.])
    y = sim_quantiles(taus, quantiles)
    w = np.ones_like(y)
    qr = QuantileRegressor(taus[1:-1], solver_options=dict(verbose=True)).fit(X, y, w)
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
    score = qr.score(X, y, w)
    assert score > .25

if __name__ == '__main__':
    test_quantile_regression()
    