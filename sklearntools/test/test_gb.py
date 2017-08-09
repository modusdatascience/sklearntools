import numpy as np
from sklearntools.gb import GradientBoostingEstimator
from sklearn.ensemble.gradient_boosting import QuantileLossFunction
from numpy.testing.utils import assert_approx_equal
from sklearntools.earth import Earth
from nose.tools import assert_less




def test_gradient_boosting_estimator():
    m = 10000
    n = 10
    p = .75
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    mu = np.dot(X, beta)
    y = np.random.lognormal(mu)
    model = GradientBoostingEstimator(Earth(verbose=True), QuantileLossFunction(1, p), n_estimators=50)
    model.fit(X, y)
    prediction = model.predict(X)
    q = np.mean(y <= prediction)
    print q
    assert_less(np.abs(q-p), .04)

    
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

