import numpy as np
from sklearntools.gb import GradientBoostingEstimator,\
    SmoothQuantileLossFunction, log_one_plus_exp_x, one_over_one_plus_exp_x,\
    stop_after_n_iterations_without_percent_improvement_over_threshold
from numpy.testing.utils import assert_approx_equal, assert_array_almost_equal
from sklearntools.earth import Earth
from nose.tools import assert_less, assert_greater, assert_raises, assert_true
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor,\
    QuantileLossFunction, BinomialDeviance
from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.metrics.regression import r2_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import pandas
from sklearn.datasets.samples_generator import make_classification
from nose import SkipTest
from sklearn.calibration import CalibratedClassifierCV

def test_smooth_quantile_loss_function():
    np.random.seed(0)
    n = 1000
    y1 = np.random.normal(size=n)
    y2 = np.random.normal(size=n)
    tau = .75
    alpha = .000001
    l1 = SmoothQuantileLossFunction(1, tau, alpha)
    l2 = QuantileLossFunction(1, tau)
    assert_approx_equal(l1(y1, y2) / float(n), l2(y1, y2))
    
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

def test_gradient_boosting_estimator_with_binomial_deviance_loss():
    np.random.seed(0)
    X, y = make_classification(n_classes=2)
    loss_function = BinomialDeviance(2)
    model = GradientBoostingEstimator(Earth(max_degree=2, use_fast=True, max_terms=10), loss_function)
    model.fit(X, y)
    assert_greater(np.sum(model.predict(X)==y) / float(y.shape[0]), .90)
    assert_true(np.all(0<=model.predict_proba(X)))
    assert_true(np.all(1>=model.predict_proba(X)))
#     symbols = syms(model)
#     X_ = pandas.DataFrame(X, columns=[s.name for s in symbols])
#     numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
#     y_pred_ = numpy_test_module.test_model(**X_)
#     y_pred = model.predict(X)
#     assert_array_almost_equal(y_pred, y_pred_)
    
#     numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict_proba', 'test_model'))
#     y_pred_ = numpy_test_module.test_model(**X_)
#     y_pred = model.predict_proba(X)[:,1]
#     assert_array_almost_equal(y_pred, y_pred_)

def test_with_calibrated_classifier():
    np.random.seed(0)
    X, y = make_classification(n_classes=2)
    loss_function = BinomialDeviance(2)
    model = CalibratedClassifierCV(GradientBoostingEstimator(Earth(max_degree=2, use_fast=True, max_terms=10), 
                                                                loss_function),
                                   method='isotonic')
    model.fit(X, y)
    assert_greater(np.sum(model.predict(X)==y) / float(y.shape[0]), .90)
    assert_true(np.all(0<=model.predict_proba(X)))
    assert_true(np.all(1>=model.predict_proba(X)))

@SkipTest
def test_gradient_boosting_estimator_with_smooth_quantile_loss():
    np.random.seed(0)
    m = 15000
    n = 10
    p = .8
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    mu = np.dot(X, beta)
    y = np.random.lognormal(mu)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33333333333333)
    loss_function = SmoothQuantileLossFunction(1, p, .0001)
    q_loss = QuantileLossFunction(1, p)
    model = GradientBoostingEstimator(BaggingRegressor(Earth(max_degree=2, verbose=False, use_fast=True, max_terms=10)), 
                                      loss_function, n_estimators=150, 
                                      stopper=stop_after_n_iterations_without_percent_improvement_over_threshold(3, .01), verbose=True)
    assert_raises(NotFittedError, lambda : model.predict(X_train))
    
    model.fit(X_train, y_train)
#     assert_less(len(model.estimators_)-1, 50)
#     assert_true(model.early_stop_)
    
    prediction = model.predict(X_test)
    model2 = GradientBoostingRegressor(loss='quantile', alpha=p)
    model2.fit(X_train, y_train)
    prediction2 = model2.predict(X_test)
    assert_less(q_loss(y_test, prediction), q_loss(y_test, prediction2))
    assert_greater(r2_score(y_test,prediction), r2_score(y_test,prediction2))
    q = np.mean(y_test <= prediction)
    assert_less(np.abs(q-p), .05)
    assert_greater(model.score_, 0.)
    assert_approx_equal(model.score(X_train, y_train), model.score_)
    
# def test_sym_predict():
#     np.random.seed(0)
#     m = 5000
#     n = 10
#     p = .8
#     X = np.random.normal(size=(m,n))
#     beta = np.random.normal(size=n)
#     mu = np.dot(X, beta)
#     y = np.random.lognormal(mu)
#     loss_function = SmoothQuantileLossFunction(1, p, .0001)
#     model = GradientBoostingEstimator(Earth(max_degree=1, verbose=False, use_fast=True, max_terms=10), 
#                                       loss_function, n_estimators=10)
#     model.fit(X, y)
#     symbols = syms(model)
#     X_ = pandas.DataFrame(X, columns=[s.name for s in symbols])
#     numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
#     y_pred_ = numpy_test_module.test_model(**X_)
#     y_pred = model.predict(X)
#     assert_array_almost_equal(y_pred, y_pred_)
#     
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])

