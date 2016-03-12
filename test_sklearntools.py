'''
Created on Feb 23, 2016

@author: jason
'''
import numpy as np
from sklearntools import StagedEstimator, MaskedEstimator,\
    ColumnSubsetTransformer
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from calibration import CalibratedEstimatorCV, ResponseTransformingEstimator,\
    LogTransformer, PredictorTransformer, HazardToRiskEstimator,\
    MovingAverageSmoothingEstimator, ThresholdClassifier, ProbaPredictingEstimator
from quantile import QuantileRegressor
from feature_selection import SingleEliminationFeatureImportanceEstimatorCV,\
    BackwardEliminationEstimator
from numpy.testing.utils import assert_raises
from glm import GLM
import statsmodels.api as sm
from pyearth.earth import Earth
import warnings
import pandas
warnings.simplefilter("error")

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
    score = qr.score(X, y, w)
    assert score > .25

def test_single_elimination_feature_importance_estimator_cv():
    np.random.seed(0)
    m = 100000
    n = 6
    factor = .9
    
    X = np.random.normal(size=(m,n))
    beta = 100 * np.ones(shape=n)
    for i in range(1, n):
        beta[i] = factor * beta[i-1]
    beta = np.random.permutation(beta)[:,None]
    
    y = np.dot(X, beta) + 0.01 * np.random.normal(size=(m, 1))
    
    target_sequence = np.ravel(np.argsort(beta ** 2, axis=0))
    model1 = SingleEliminationFeatureImportanceEstimatorCV(LinearRegression())
    model1.fit(X, y)
    fitted_sequence = np.ravel(np.argsort(model1.feature_importances_, axis=0))
    
    np.testing.assert_array_equal(fitted_sequence, target_sequence)
    
def test_backward_elimination_estimation():
    np.random.seed(0)
    m = 100000
    n = 6
    factor = .9
    
    X = np.random.normal(size=(m,n))
    beta = 100 * np.ones(shape=n)
    for i in range(1, n):
        beta[i] = factor * beta[i-1]
    beta = np.random.permutation(beta)[:,None]
#     beta = np.random.normal(size=(n,1))
    
    y = np.dot(X, beta) + 0.01 * np.random.normal(size=(m, 1))
    
    target_sequence = np.ravel(np.argsort(beta ** 2, axis=0))
    model1 = BackwardEliminationEstimator(SingleEliminationFeatureImportanceEstimatorCV(LinearRegression(), check_constant_model=False))
    model1.fit(X, y)
    
#     model2 = BRFE(FeatureImportanceEstimatorCV(LinearRegression()))
#     model2.fit(X, y)
    
    np.testing.assert_array_equal(model1.elimination_sequence_, target_sequence)

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
        
    model = MaskedEstimator(LinearRegression(), [True, False]) & MaskedEstimator(ProbaPredictingEstimator(LogisticRegression()), [False, True])
#     MultipleResponseEstimator([('linear', np.array([True, False], dtype=bool), LinearRegression()), 
#                                        ('logistic', np.array([False, True], dtype=bool), ProbaPredictingEstimator(LogisticRegression()))])
    model.fit(X, y)
    
    assert np.mean(beta1 - model.estimators_[0].estimator_.coef_) < .01
    assert np.mean(beta2 - model.estimators_[1].estimator_.estimator_.coef_) < .01
    model.get_params()
    model.predict(X)

def test_calibration():
    np.random.seed(1)
    m = 10000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    y_lin = np.dot(X, beta)
    y_clas = np.random.binomial( 1, 1. / (1. + np.exp(-y_lin)) )
    y = np.concatenate([y_lin, y_clas], axis=1)
    estimator = MaskedEstimator(LinearRegression(), np.array([True, False], dtype=bool))
    calibrator = MaskedEstimator(LogisticRegression(), [False, True])
#     estimator = linear_regressor & calibrator
#     MultipleResponseEstimator([('estimator', np.array([True, False], dtype=bool), LinearRegression())])
#     calibrator = MultipleResponseEstimator([('calibrator', np.array([False, True], dtype=bool), LogisticRegression())])
    model = CalibratedEstimatorCV(estimator, calibrator)
    model.fit(X, y)
    assert np.max(beta[:, 0] - model.estimator_.estimator_.coef_) < .000001
    assert np.max(model.calibrator_.estimator_.coef_ - 1.) < .1

def test_predictor_transformer_calibration():
    np.random.seed(1)
    m = 10000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    y_lin = np.dot(X, beta)
    y_clas = np.random.binomial( 1, 1. / (1. + np.exp(-y_lin)) )
    y = np.concatenate([y_lin, y_clas], axis=1)
    estimator = MaskedEstimator(LinearRegression(), np.array([True, False], dtype=bool))
    calibrator = MaskedEstimator(LogisticRegression(), [False, True])
#     estimator = linear_regressor & calibrator
#     MultipleResponseEstimator([('estimator', np.array([True, False], dtype=bool), LinearRegression())])
#     calibrator = MultipleResponseEstimator([('calibrator', np.array([False, True], dtype=bool), LogisticRegression())])
    model = PredictorTransformer(estimator) >> calibrator
    model.fit(X, y)
    assert np.max(beta[:, 0] - model.intermediate_stages_[0].estimator_.estimator_.coef_) < .000001
    assert np.max(model.final_stage_.estimator_.coef_ - 1.) < .1
    
def test_pipeline():
    np.random.seed(1)
    m = 10000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    beta[np.random.binomial(p=2.0/float(n), n=1, size=n).astype(bool)] = 0
    y = np.dot(X, beta) + 0.5 * np.random.normal(size=(m, 1))
    beta_reduced = beta[beta != 0]
    
    model = BackwardEliminationEstimator(SingleEliminationFeatureImportanceEstimatorCV(LinearRegression(), check_constant_model=False)) 
    model >>= LinearRegression()
    
    model.fit(X, y)
    assert np.max(np.abs(model.final_stage_.coef_ - beta_reduced)) < .1

def test_response_transforming_estimator():
    np.random.seed(1)
    m = 10000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    sigma = .1
    y_pre = np.dot(X, beta) + sigma * np.random.normal(size=(m,1))
    y_post = np.exp(y_pre)
    
    model = ResponseTransformingEstimator(LinearRegression(), LogTransformer(offset=0.))
    model.fit(X, y_post)
    
    assert np.abs(np.mean(model.predict(X) - y_pre)) < .01
    
    # Because LinearRegression has no transform method
    assert_raises(AttributeError, lambda: model.transform(X))
    
def test_hazard_to_risk():
    np.random.seed(1)
    
    m = 10000
    n = 10
    
    # Simulate an event under constant hazard, with hazard = X * beta and 
    # iid exponentially distributed exposure times.
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    hazard = np.exp(np.dot(X, beta))
    exposure = np.random.exponential(size=(m,1))
    rate = np.random.poisson(hazard * exposure) / exposure
    
    model = CalibratedEstimatorCV(GLM(sm.families.Gaussian(sm.families.links.log), add_constant=False), 
                                  ProbaPredictingEstimator(ThresholdClassifier(HazardToRiskEstimator(LogisticRegression()))))
    
    model.fit(X, rate, exposure=exposure)
    
    y_pred = model.predict(X, exposure)
    assert np.abs((np.sum(y_pred) - np.sum(rate > 0)) / np.sum(rate > 0))  < .1
    assert np.max(np.abs(model.estimator_.coef_ - beta[:,0])) < .1

def test_hazard_to_risk_staged():
    np.random.seed(1)
    
    m = 10000
    n = 10
    
    # Simulate an event under constant hazard, with hazard = X * beta and 
    # iid exponentially distributed exposure times.
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    hazard = np.exp(np.dot(X, beta))
    exposure = np.random.exponential(size=(m,1))
    rate = np.random.poisson(hazard * exposure) / exposure
    
    model = CalibratedEstimatorCV(GLM(sm.families.Gaussian(sm.families.links.log), add_constant=False), 
                                  ProbaPredictingEstimator(ThresholdClassifier(HazardToRiskEstimator(LogisticRegression()))))
    
    model.fit(X, rate, exposure=exposure)
    
    y_pred = model.predict(X, exposure)
    assert np.abs((np.sum(y_pred) - np.sum(rate > 0)) / np.sum(rate > 0))  < .1
    assert np.max(np.abs(model.estimator_.coef_ - beta[:,0])) < .1

def test_moving_average_smoothing_estimator():
    np.random.seed(1)
    
    m = 10000
    n = 10
    
    # Simulate an event under constant hazard, with hazard = X * beta and 
    # iid exponentially distributed exposure times.
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    hazard = np.exp(np.dot(X, beta))
    exposure = np.random.exponential(size=(m,1))
    rate = np.random.poisson(hazard * exposure) / exposure
    
    model = CalibratedEstimatorCV(GLM(sm.families.Gaussian(sm.families.links.log), add_constant=False), 
                                  ThresholdClassifier(HazardToRiskEstimator(MovingAverageSmoothingEstimator(Earth()))))
    
    model.fit(X, rate, exposure=exposure)
    
    y_pred = model.predict(X, exposure)
    assert np.abs((np.sum(y_pred) - np.sum(rate > 0)) / np.sum(rate > 0))  < .1
    assert np.max(np.abs(model.estimator_.coef_ - beta[:,0])) < .1

def test_staged_estimator():
    np.random.seed(1)
    m = 10000
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n,1))
    beta[np.random.binomial(p=2.0/float(n), n=1, size=n).astype(bool)] = 0
    y = np.dot(X, beta) + 0.5 * np.random.normal(size=(m, 1))
    beta_reduced = beta[beta != 0]
    
    stage0 = BackwardEliminationEstimator(SingleEliminationFeatureImportanceEstimatorCV(LinearRegression(), check_constant_model=False)) 
    stage1 = LinearRegression()
    model = StagedEstimator([stage0, stage1])
    
    model.fit(X, y)
    assert np.max(np.abs(model.final_stage_.coef_ - beta_reduced)) < .1
#     
#     y_lin = np.dot(X, beta)
#     y_clas = np.random.binomial( 1, 1. / (1. + np.exp(-y_lin)) )
#     y = np.concatenate([y_lin, y_clas], axis=1)
#     estimator = mask_estimator(LinearRegression(), np.array([True, False], dtype=bool))
#     calibrator = mask_estimator(LogisticRegression(), [False, True])
# #     estimator = linear_regressor & calibrator
# #     MultipleResponseEstimator([('estimator', np.array([True, False], dtype=bool), LinearRegression())])
# #     calibrator = MultipleResponseEstimator([('calibrator', np.array([False, True], dtype=bool), LogisticRegression())])
#     model = CalibratedEstimatorCV(estimator, calibrator)
#     model.fit(X, y)
#     assert np.max(beta[:, 0] - model.estimator_.estimators_[0][2].coef_) < .000001
#     assert np.max(model.calibrator_.estimators_[0][2].coef_ - 1.) < .1

def test_column_subset_transformer():
    m = 1000
    n = 10
    X = np.random.normal(size=(m,n))
    x_cols = [0,3,4,5]
    y_cols = 9
    sample_weight_cols = 8
    exposure_cols = 7
    
    subsetter1 = ColumnSubsetTransformer(x_cols=x_cols, y_cols=y_cols, 
                                         sample_weight_cols=sample_weight_cols,
                                         exposure_cols=exposure_cols)
    np.testing.assert_array_equal(subsetter1.transform(X), X[:, x_cols])
    args = {'X': X}
    subsetter1.update(args)
    np.testing.assert_array_equal(args['X'], X[:, x_cols])
    np.testing.assert_array_equal(args['y'], X[:, y_cols])
    np.testing.assert_array_equal(args['sample_weight'], X[:, sample_weight_cols])
    np.testing.assert_array_equal(args['exposure'], X[:, exposure_cols])
    
    X_ = pandas.DataFrame(X, columns=['x%d' % n for n in range(10)])
    x_cols_ = ['x%d' % n for n in x_cols]
    y_cols_ = 'x%d' % y_cols
    sample_weight_cols_ = 'x%d' % sample_weight_cols
    exposure_cols_ = 'x%d' % exposure_cols
    subsetter2 = ColumnSubsetTransformer(x_cols=x_cols_, y_cols=y_cols_, 
                                         sample_weight_cols=sample_weight_cols_,
                                         exposure_cols=exposure_cols_)
    np.testing.assert_array_equal(subsetter2.transform(X_), X[:, x_cols])
    args_ = {'X': X_}
    subsetter2.update(args_)
    np.testing.assert_array_equal(args_['X'], X[:, x_cols])
    np.testing.assert_array_equal(args_['y'], X[:, y_cols])
    np.testing.assert_array_equal(args_['sample_weight'], X[:, sample_weight_cols])
    np.testing.assert_array_equal(args_['exposure'], X[:, exposure_cols])
    
    lin = ColumnSubsetTransformer(x_cols=x_cols_, y_cols=y_cols_) >> LinearRegression()
    lin.fit(X_)
    lin.predict(X_.loc[:, x_cols_])
    lin.score(X_)
    
if __name__ == '__main__':
#     test_quantile_regression()
#     test_single_elimination_feature_importance_estimator_cv()
#     test_backward_elimination_estimation()
#     test_multiple_response_regressor()
#     test_calibration()
#     test_predictor_transformer_calibration()
#     test_pipeline()
#     test_response_transforming_estimator()
#     test_hazard_to_risk()
#     test_moving_average_smoothing_estimator()
#     test_staged_estimator()
    test_column_subset_transformer()
    print 'Success!'
    
    