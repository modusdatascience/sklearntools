from sklearn.datasets.base import load_boston
from sklearntools.super_learner import SuperLearner, OrderTransformer,\
    NonNegativeLeastSquaresRegressor
from sklearn.linear_model.base import LinearRegression
from pyearth import Earth
from sklearn.metrics.regression import r2_score
from sklearntools.kfold import CrossValidatingEstimator
import numpy as np
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.utility import exec_module
import pandas
from numpy.ma.testutils import assert_array_almost_equal
from toolz.itertoolz import first
import os
from shutil import rmtree
from sklearn.externals.joblib import Memory
from xgboost.sklearn import XGBRegressor
from nose.tools import assert_equal
from sklearntools.transformers import Identity, TransformingEstimator,\
    VariableTransformer
from sklearn2code.renderers import numpy_flat

def test_super_learner():
    np.random.seed(0)
    X, y = load_boston(return_X_y=True)
    X = pandas.DataFrame(X, columns=['x%d'%i for i in range(X.shape[1])])
    model = CrossValidatingEstimator(SuperLearner([('linear', LinearRegression()), ('earth', Earth(max_degree=2))],
                         LinearRegression(), cv=5, n_jobs=1), cv=5)
    cv_pred = model.fit_predict(X, y)
    pred = model.predict(X)
    cv_r2 = r2_score(y, cv_pred)
    best_component_cv_r2 = max([r2_score(y, first(model.estimator_.cross_validating_estimators_.values()).cv_predictions_) for i in range(2)])
    assert cv_r2 >= .9*best_component_cv_r2
    
    code = sklearn2code(model, ['predict'], numpy_flat)
    module = exec_module('module', code)
    test_pred = module.predict(**X)
    try:
        assert_array_almost_equal(np.ravel(pred), np.ravel(test_pred))
    except:
        idx = np.abs(np.ravel(pred) - np.ravel(test_pred)) > .000001
        print(np.ravel(pred)[idx])
        print(np.ravel(test_pred)[idx])
        raise
    print(r2_score(y, pred))
    print(r2_score(y, cv_pred))
    
    print(max([r2_score(y, first(model.estimator_.cross_validating_estimators_.values()).cv_predictions_) for i in range(2)]))

def test_super_learner_with_memory():
    memory_dir = 'test_memory_dir'
    if os.path.exists(memory_dir):
        rmtree(memory_dir)
    
    X, y = load_boston(return_X_y=True)
    try:
        model = SuperLearner([('linear', LinearRegression()), ('earth', Earth(max_degree=2))],
                         LinearRegression(), cv=5, n_jobs=1, memory=Memory(memory_dir, verbose=0))
        model.fit(X, y)
        
        assert all([not est.loaded_from_cache_ for est in model.cross_validating_estimators_.values()])
        
        model2 = SuperLearner([('linear', LinearRegression()), ('earth', Earth(max_degree=2))],
                         LinearRegression(), cv=5, n_jobs=1, memory=Memory(memory_dir, verbose=0))
        
        model2.fit(X, y)
        assert all([est.loaded_from_cache_ for est in model2.cross_validating_estimators_.values()])
        
        model3 = SuperLearner([('linear', LinearRegression()), ('earth', Earth(max_degree=2)), ('xgb', XGBRegressor())],
                         LinearRegression(), cv=5, n_jobs=1, memory=Memory(memory_dir, verbose=0))
        
        model3.fit(X, y)
        assert model3.cross_validating_estimators_['linear'].loaded_from_cache_
        assert model3.cross_validating_estimators_['earth'].loaded_from_cache_
        assert not model3.cross_validating_estimators_['xgb'].loaded_from_cache_
        
    finally:
        if os.path.exists(memory_dir):
            rmtree(memory_dir)
        
def test_order_transformer():
    X, y = load_boston(return_X_y=True)
    model = OrderTransformer()
    XX = X.copy()
    model.fit(X, y)
    O = model.transform(X)
    for i in range(X.shape[0]):
        assert_equal(set(XX[i,:]), set(O.iloc[i,:]))
        assert_equal(list(O.iloc[i,:]), list(sorted(O.iloc[i,:])))


def test_memorization_with_complicated_model():
    def identities(*args):
        return {arg: Identity(arg) for arg in args}
    
    def create_model(*inputs):
        standard_model_inputs = inputs
        return TransformingEstimator(
            XGBRegressor(),
            y_transformer = VariableTransformer(identities('x5'),
                                                strict=True, exclusive=True),
            x_transformer = VariableTransformer(identities(*standard_model_inputs),
                                                strict=True, exclusive=True),
            )
    m = 1000
    n = 10
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    beta[5] = 0.
    X[:, 5] = np.random.normal(np.dot(X, beta))
    X = pandas.DataFrame(X, columns=['x%d' % i for i in range(n)])
    
    memory_dir = 'test_memory_dir'
    if os.path.exists(memory_dir):
        rmtree(memory_dir)
    try:
        model1 = SuperLearner([('x1', create_model('x1')), ('x2-4', create_model('x2', 'x3', 'x4'))],
                              NonNegativeLeastSquaresRegressor(normalize_coefs=False),
                              y_transformer=VariableTransformer(identities('x5'),
                                                strict=True, exclusive=True),
                              memory=Memory(memory_dir, verbose=0)
                              )
        model1.fit(X)
        assert all([not est.loaded_from_cache_ for est in model1.cross_validating_estimators_.values()])
        model2 = SuperLearner([('x1', create_model('x1')), ('x2-4', create_model('x2', 'x3', 'x4'))],
                              NonNegativeLeastSquaresRegressor(normalize_coefs=False),
                              y_transformer=VariableTransformer(identities('x5'),
                                                strict=True, exclusive=True),
                              memory=Memory(memory_dir, verbose=0)
                              )
        model2.fit(X)
        assert all([est.loaded_from_cache_ for est in model2.cross_validating_estimators_.values()])
    finally:
        if os.path.exists(memory_dir):
            rmtree(memory_dir)


def test_non_negative_least_squares_regressor():
    m = 1000
    n = 10
    X = np.random.normal(size=(m,n))
    beta = np.random.uniform(0., 1., size=n)
    y = np.random.normal(np.dot(X, beta))
    X = pandas.DataFrame(X, columns=['x%d'%i for i in range(X.shape[1])])
    
    model = NonNegativeLeastSquaresRegressor(normalize_coefs=False)
    model.fit(X, y)
    
    pred = model.predict(X)
    code = sklearn2code(model, ['predict'], numpy_flat)
    module = exec_module('module', code)
    test_pred = module.predict(**X)
    try:
        assert_array_almost_equal(np.ravel(pred), np.ravel(test_pred))
    except:
        idx = np.abs(np.ravel(pred) - np.ravel(test_pred)) > .000001
        print(np.ravel(pred)[idx])
        print(np.ravel(test_pred)[idx])
        raise


def test_order_transformer_export():
    m = 1000
    n = 10
    X = np.random.normal(size=(m,n))
    X = pandas.DataFrame(X, columns=['x%d'%i for i in range(X.shape[1])])
    
    model = OrderTransformer()
    model.fit(X)
    trans = model.transform(X)
    
    code = sklearn2code(model, ['transform'], numpy_flat)
    module = exec_module('module', code)
    test_trans = np.stack(module.transform(**X), 1)
    assert_array_almost_equal(trans, test_trans)

    
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])
