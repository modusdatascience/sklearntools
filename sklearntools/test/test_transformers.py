from sklearntools.transformers import Constant, VariableTransformer, Identity,\
    Censor, NanMap, Log, TransformingEstimator
import numpy as np
import pandas
from numpy.testing.utils import assert_array_almost_equal
from sklearn.datasets.base import load_boston
from pyearth.earth import Earth
from sklearntools.calibration import ResponseTransformingEstimator
from sklearn.metrics.regression import r2_score
from sklearn.linear_model.base import LinearRegression
from sklearn2code.sklearn2code import sklearn2code
from sklearn2code.renderers import numpy_flat
from sklearn2code.utility import exec_module
# from sklearntools.sym.printers import exec_module, model_to_code

def test_with_response_transformation():
    X, y = load_boston(return_X_y=True)
    
    log_y = np.log(y)
    
    X = pandas.DataFrame(X, columns=['x%d' % i for i in range(X.shape[1])])
    y = pandas.DataFrame(y, columns=['y'])
    
    
    transformer = VariableTransformer(dict(y=Log(Identity('y'))))
    model = ResponseTransformingEstimator(Earth(), transformer)
    model.fit(X, y)
    log_y_pred = model.predict(X)
    assert r2_score(log_y, log_y_pred) > .8
    assert r2_score(y, log_y_pred) < .1
    
    
def test_transformation_system():
    np.random.seed(1)
    x = Identity('x')
    y = Identity('y')
    z = Identity('z')
    
    d = (x + y) / z
    
    transformer = VariableTransformer(dict(d=d), exclusive=True)
    X = pandas.DataFrame(np.random.normal(size=(10,3)), columns=['x','y','z'])
    transformer.fit(X)
    assert_array_almost_equal(transformer.transform(X)['d'], (X['x'] + X['y']) / X['z'])
#     numpy_test_module = exec_module('numpy_test_module', model_to_code(transformer, 'numpy', 'transform', 'test_model'))
#     assert_array_almost_equal(pandas.DataFrame(dict(zip(['x', 'y', 'z', 'd'], numpy_test_module.test_model(**X))))[['x', 'y', 'z', 'd']], transformer.transform(X))

def test_prediction_multiplier_transformer():
    n = 100
    np.random.seed(1)
    x = Identity('x')
    y = Identity('y')
    z = Identity('z')
    
    d = x + y + Constant(2) * z
    
    model = TransformingEstimator(
                                  estimator=LinearRegression(), 
                                  y_transformer = VariableTransformer(dict(d=d), exclusive=True),
                                  prediction_multiplier_transformer = VariableTransformer(dict(m = 
                                                                                               Censor(Constant(1), z == Constant(0))
                                                                                               ),
                                                                                          exclusive=True)
                                  )
    
    X = pandas.DataFrame(np.random.normal(size=(n,3)), columns=['x','y','z'])
    X['z'] *= np.random.binomial(1, .5, size=n)
    model.fit(X)
    pred = model.predict(X)
    assert_array_almost_equal(np.where(np.isnan(np.ravel(pred))), np.where(X['z'] == 0))
    assert_array_almost_equal(model.estimator_.predict(X)[np.where(~np.isnan(pred))],
                              pred[~np.isnan(pred)])
    
    code = sklearn2code(model, ['predict'], numpy_flat)
    module = exec_module('module', code)
    test_pred = module.predict(**X)
    assert_array_almost_equal(np.ravel(pred), test_pred)
    
                                                                                          
    
def test_rate():
    np.random.seed(1)
    X = pandas.DataFrame({'count': np.random.poisson(1., size=100), 'duration': np.random.poisson(5., size=100)})
    rate = Censor(Identity('count') / Identity('duration'), Identity('duration') < 4)
    transformer = VariableTransformer(dict(rate=rate))
    transformer.fit(X)
    target = X['count'] / X['duration']
    target[X['duration'] < 4] = np.nan
    assert_array_almost_equal(transformer.transform(X)['rate'], target)
#     numpy_test_module = exec_module('numpy_test_module', model_to_code(transformer, 'numpy', 'transform', 'test_model'))
#     assert_array_almost_equal(pandas.DataFrame(dict(zip(['count', 'duration', 'rate'], numpy_test_module.test_model(**X))))[['count', 'duration', 'rate']], transformer.transform(X))

def test_uncensor():
    X = pandas.DataFrame(np.random.normal(size=(10,3)), columns=['x','y','z'])
    X.loc[1,'x'] = np.nan
    X.loc[2, 'y'] = np.nan
    transformer = NanMap({'x': 100.})
    transformer.fit(X)
    X_ = transformer.transform(X)
    assert_array_almost_equal(X['y'], X_['y'])
    assert not (X['x'] == X_['x']).all()
    fix = X['x'].copy()
    fix[1] = 100.
    assert_array_almost_equal(fix, X_['x'])

def test_non_strict():
    X = pandas.DataFrame(np.random.normal(size=(10,3)), columns=['x','y','z'])
    X.loc[1,'x'] = np.nan
    X.loc[2, 'y'] = np.nan
    transformer = NanMap({'x': 100.,
                          'w': 0.})
    transformer.fit(X)
    X_ = transformer.transform(X)
    assert_array_almost_equal(X['y'], X_['y'])
    assert not (X['x'] == X_['x']).all()
    fix = X['x'].copy()
    fix[1] = 100.
    assert_array_almost_equal(fix, X_['x'])

if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])
