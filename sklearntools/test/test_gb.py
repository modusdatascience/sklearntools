import numpy as np
from sklearntools.gb import GradientBoostingEstimator,\
    SmoothQuantileLossFunction, log_one_plus_exp_x, one_over_one_plus_exp_x
from numpy.testing.utils import assert_approx_equal, assert_array_almost_equal
from sklearntools.earth import Earth
from nose.tools import assert_less, assert_greater

def test_log_one_plus_exp_x():
    x = np.arange(-20.,100.)
    y_1 = np.log(1+np.exp(x))
    y_2 = log_one_plus_exp_x(x)
    assert_array_almost_equal(y_1, y_2)

def test_one_over_one_plus_exp_x():
    x = np.arange(-20.,100.)
    y_1 = 1. / (1. + np.exp(x))
    y_2 = one_over_one_plus_exp_x(x)
    assert_array_almost_equal(y_1, y_2)

def test_gradient_boosting_estimator():
    np.random.seed(1)
    m = 10000
    n = 10
    p = .5
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    mu = np.dot(X, beta)
    y = np.random.lognormal(mu)
    model = GradientBoostingEstimator(Earth(verbose=True), SmoothQuantileLossFunction(1, p, .01), n_estimators=100)
    model.fit(X, y)
    prediction = model.predict(X)
    q = np.mean(y <= prediction)
    print q
    print model.score_
    assert_less(np.abs(q-p), .05)
    assert_greater(model.score_, 0.)
    assert_approx_equal(model.score(X, y), model.score_)
    

    
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

