import numpy as np
from sklearntools.gb import GradientBoostingEstimator,\
    SmoothQuantileLossFunction, log_one_plus_exp_x, one_over_one_plus_exp_x
from numpy.testing.utils import assert_approx_equal, assert_array_almost_equal
from sklearntools.earth import Earth
from nose.tools import assert_less, assert_greater
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor,\
    QuantileLossFunction
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.metrics.regression import r2_score
from sklearn.tree.tree import DecisionTreeRegressor
from nose import SkipTest
from distutils.version import LooseVersion
import sklearn
from types import MethodType
from sklearn.cross_validation import train_test_split

# Patch over bug in scikit learn (issue #9539)
if LooseVersion(sklearn.__version__) <= LooseVersion('0.18.2'):
    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        diff = y - pred
        alpha = self.alpha
    
        mask = y > pred
        if sample_weight is None:
            loss = (alpha * diff[mask].sum() -
                    (1.0 - alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            loss = ((alpha * np.sum(sample_weight[mask] * diff[mask]) -
                    (1.0 - alpha) * np.sum(sample_weight[~mask] * diff[~mask])) /
                    sample_weight.sum())
        return loss
    QuantileLossFunction.__call__ = MethodType(__call__, None, QuantileLossFunction)
    
def test_smooth_quantile_loss_function():
    np.random.seed(0)
    n = 1000
    y1 = np.random.normal(size=n)
    y2 = np.random.normal(size=n)
    tau = .75
    alpha = .000001
    l1 = SmoothQuantileLossFunction(1, tau, alpha)
    l2 = QuantileLossFunction(1, tau)
    print 'l1 =', l1(y1, y2)
    print 'l2 =', l2(y1, y2)
    assert_approx_equal(l1(y1, y2), l2(y1, y2))
    
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
    np.random.seed(0)
    m = 15000
    n = 10
    p = .8
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    mu = np.dot(X, beta)
    y = np.random.lognormal(mu)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    loss_function = SmoothQuantileLossFunction(1, p, .0001)
    q_loss = QuantileLossFunction(1, p)
    model = GradientBoostingEstimator(BaggingRegressor(Earth(max_degree=2, verbose=True, use_fast=True, max_terms=10)), 
                                      loss_function, n_estimators=150)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    model2 = GradientBoostingRegressor(loss='quantile', alpha=p)
    model2.fit(X_train, y_train)
    prediction2 = model2.predict(X_test)
    model_loss = loss_function(y_test, prediction)
    model2_loss = loss_function(y_test, prediction2)
    
    print 'loss1 =', model_loss
    print 'loss2 =', model2_loss
    print 'qloss1 =', q_loss(y_test, prediction)
    print 'qloss2 =', q_loss(y_test, prediction2)
    print 'r2_score_1 =', r2_score(y_test,prediction)
    print 'r2_score_2 =', r2_score(y_test,prediction2) 
    q = np.mean(y_test <= prediction)
    q2 = np.mean(y_test <= prediction2)
    print q, q2
    print model.score_
    assert_less(np.abs(q-p), .05)
    assert_greater(model.score_, 0.)
    assert_approx_equal(model.score(X_train, y_train), model.score_)
    

    
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

