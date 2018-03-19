import numpy as np
from sklearntools.earth import Earth
import pandas
from sklearntools.sym.printers import model_to_code, exec_module
from sklearntools.sym.sym_predict import sym_predict
from numpy.ma.testutils import assert_array_almost_equal
import execjs
from sklearntools.calibration import LogTransformer,\
    ResponseTransformingEstimator, CalibratedEstimatorCV, \
    MovingAverageSmoothingEstimator, SelectorTransformer, IntervalTransformer, \
    PredictorTransformer, ProbaPredictingEstimator
from sklearntools.kfold import CrossValidatingEstimator
from sklearn.tree.tree import DecisionTreeRegressor
from nose.tools import assert_almost_equal
from sklearntools.sym.syms import syms
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier,\
    LeastSquaresError, LeastAbsoluteError, HuberLossFunction,\
    GradientBoostingRegressor
from sklearntools.sym.sym_predict_proba import sym_predict_proba
from sklearntools.gb import GradientBoostingEstimator
from nose import SkipTest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model.logistic import LogisticRegression
from sympy.core.symbol import Symbol

def test_sklearn_gradient_boosting_classifier_export():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    X = np.random.normal(size=(m,10))
    thresh = np.random.normal(size=10)
    X_transformed = X * (X > thresh)
    beta = np.random.normal(size=10)
    y = (np.dot(X_transformed, beta) + np.random.normal(size=m)) > 0
    
    # Train a gradient boosting classifier
    model = GradientBoostingClassifier(max_depth=10, n_estimators=10)
    model.fit(X, y)
    
    # Export python code and check output
    y_pred = model.predict_proba(X)[:,1]
    X_ = pandas.DataFrame(X, columns=[s.name for s in syms(model)])
    code = model_to_code(model, 'numpy', 'predict_proba', 'test_model')
    numpy_test_module = exec_module('numpy_test_module', code)
    y_pred_numpy = numpy_test_module.test_model(**X_)
    assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))

def test_least_squares_error_export():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    X = np.random.normal(size=(m,10))
    beta = np.random.normal(size=10)
    y = np.random.normal(np.dot(X, beta))
    
    # Fit a model
    model = GradientBoostingEstimator(Earth(max_terms=5), loss_function=LeastSquaresError(1))
    model.fit(X, y)
    
    # Export as sympy expression
    expr = sym_predict(model)
    
    # Check some values
    y_pred = model.predict(X)
    X_ = pandas.DataFrame(X, columns=[s.name for s in syms(model)])
    for i in range(10):
        row = dict(X_.iloc[i,:])
        assert_almost_equal(y_pred[i], expr.evalf(16, row))
    
    # Export python code and check output
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred_numpy = numpy_test_module.test_model(**X_)
    assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))

def test_least_absolute_error_export():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    X = np.random.normal(size=(m,10))
    beta = np.random.normal(size=10)
    y = np.random.normal(np.dot(X, beta))
    
    # Fit a model
    model = GradientBoostingEstimator(Earth(max_terms=5), loss_function=LeastAbsoluteError(1))
    model.fit(X, y)
    
    # Export as sympy expression
    expr = sym_predict(model)
    
    # Check some values
    y_pred = model.predict(X)
    X_ = pandas.DataFrame(X, columns=[s.name for s in syms(model)])
    for i in range(10):
        row = dict(X_.iloc[i,:])
        assert_almost_equal(y_pred[i], expr.evalf(16, row))
    
    # Export python code and check output
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred_numpy = numpy_test_module.test_model(**X_)
    assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))

def test_huber_loss_export():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    X = np.random.normal(size=(m,10))
    beta = np.random.normal(size=10)
    y = np.random.normal(np.dot(X, beta))
    
    # Fit a model
    model = GradientBoostingEstimator(Earth(max_terms=5, smooth=True), loss_function=HuberLossFunction(1))
    model.fit(X, y)
    
    # Export as sympy expression
    expr = sym_predict(model)
    
    # Check some values
    y_pred = model.predict(X)
    X_ = pandas.DataFrame(X, columns=[s.name for s in syms(model)])
    for i in range(10):
        row = dict(X_.iloc[i,:])
        assert_almost_equal(y_pred[i], expr.evalf(16, row))
    
    # Export python code and check output
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred_numpy = numpy_test_module.test_model(**X_)
    assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))

def test_gradient_boosting_regressor():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    X = np.random.normal(size=(m,10))
    beta = np.random.normal(size=10)
    y = np.random.normal(np.dot(X, beta))
    
    # Fit a model
    model = GradientBoostingRegressor(n_estimators=10)
    model.fit(X, y)
    
    # Export as sympy expression
    expr = sym_predict(model)
    
    # Check some values
    y_pred = model.predict(X)
    X_ = pandas.DataFrame(X, columns=[s.name for s in syms(model)])
    for i in range(10):
        row = dict(X_.iloc[i,:])
        assert_almost_equal(y_pred[i], expr.evalf(16, row))
    
    # Export python code and check output
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred_numpy = numpy_test_module.test_model(**X_)
    assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))

def test_decision_tree_export():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    X = np.random.normal(size=(m,10))
    thresh = np.random.normal(size=10)
    X_transformed = X * (X > thresh)
    beta = np.random.normal(size=10)
    y = np.dot(X_transformed, beta) + np.random.normal(size=m)
    
    # Train a decision tree regressor
    model = DecisionTreeRegressor(max_depth=10)
    model.fit(X, y)
    
    # Export as sympy expression
    expr = sym_predict(model)
    
    # Check some values
    y_pred = model.predict(X)
    X_ = pandas.DataFrame(X, columns=[s.name for s in syms(model)])
    for i in range(10):
        row = dict(X_.iloc[i,:])
        assert_almost_equal(y_pred[i], expr.evalf(16, row))
    
    # Export python code and check output
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred_numpy = numpy_test_module.test_model(**X_)
    assert_array_almost_equal(np.ravel(y_pred_numpy), np.ravel(y_pred))
    
def test_sympy_export():
    np.random.seed(1)
    m = 1000
    n = 10
    X = np.random.normal(scale=.5,size=(m,n))**2
    beta = np.random.normal(scale=1.5,size=n)**2
    eta = np.dot(X, beta)
    missing = np.random.binomial(p=.5, n=1, size=(m,n)) == 1
    X[missing] = None
    X = pandas.DataFrame(X, columns=['col%d' % i for i in range(n)])
    y = np.random.exponential(eta)
#     y = np.random.binomial(1, 1. / (1. + np.exp(-eta)))
    
    model = ResponseTransformingEstimator(estimator = Earth(allow_missing=True, max_terms=10),
                                          transformer = LogTransformer())
    model >>= Earth(allow_missing=True, max_terms=10, verbose=False) 
    model = CalibratedEstimatorCV(estimator=model, 
                                    calibrator=MovingAverageSmoothingEstimator(estimator=Earth(), window_size=10))
#     ProbaPredictingEstimator(ThresholdClassifier(LogisticRegression()))
    model = CrossValidatingEstimator(estimator=model)
    model.fit(X, y)
    
#     print model_to_code(model, 'numpy', 'predict', 'test_model')
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred = numpy_test_module.test_model(**X)
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))
    
    python_test_module = exec_module('python_test_module', model_to_code(model, 'python', 'predict', 'test_model'))
    y_pred = [python_test_module.test_model(**row) for i, row in X.iterrows()]
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))
    
    # Skip the javascript part for now
    return
    
    js = execjs.get(execjs.runtime_names.PyV8)
#     print model_to_code(model, 'javascript', 'predict', 'test_model')
    context = js.compile(model_to_code(model, 'javascript', 'predict', 'test_model'))
    y_pred = [context.eval('test_model(col3=%s, col8=%s)' % (str(row['col3']) if not np.isnan(row['col3']) else 'NaN', 
                                                             str(row['col8']) if not np.isnan(row['col8']) else 'NaN')) 
              for i, row in X.iloc[:10,:].iterrows()]
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X.iloc[:10,:])))

def test_more_sym_stuff():
    np.random.seed(1)
    m = 1000
    n = 10
    X = np.random.normal(size=(m,n))
    X_bin = X > 0
    X[:,1] = X[:,1] ** 2
    X_bin[:,1] = np.log(X[:,1] + 1)
    beta = np.random.normal(size=n)
    eta = np.dot(X_bin, beta)
    y = eta + 0.1 * np.random.normal(size=m)
    cols = map(lambda i: 'x%d'%i, range(n))
    X = pandas.DataFrame(X, columns=cols)
    
    model = (SelectorTransformer(['x1']) >> LogTransformer()) & (IntervalTransformer(lower=0., lower_closed=False) & SelectorTransformer(cols))
    model >>= (Earth() & Earth())
    model >>= Earth()
    model = PredictorTransformer(model)
    model.fit(X, y)
    
#     print sym_predict(model)
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred = numpy_test_module.test_model(**X)
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))

@SkipTest
def test_sym_predict_prior_probability_estimator():
    np.random.seed(1)
    m = 1000
    n = 10
    X = np.random.normal(scale=.5,size=(m,n))**2
    beta = np.random.normal(scale=1.5,size=n)**2
    eta = np.dot(X, beta)
    X = pandas.DataFrame(X, columns=['col%d' % i for i in range(n)])
    p = 1. / (1. + np.exp(-eta))
    y = np.random.binomial(3, p)
    
    model = ProbaPredictingEstimator(GradientBoostingClassifier(n_estimators=100))
    
    model.fit(X, y)
    
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred = numpy_test_module.test_model(**X)
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))
    

def test_sym_predict_calibrated_classifier_cv():
    np.random.seed(1)
    
    # Create some data
    m = 10000
    n = 10
    X = np.random.normal(size=(m,n))
    thresh = np.random.normal(size=n)
    X_transformed = X * (X > thresh)
    beta = np.random.normal(size=n)
    y = (np.dot(X_transformed, beta) + np.random.normal(size=m)) > 0
    X = pandas.DataFrame(X, columns=['x%d' % i for i in range(n)])
    
    model = ProbaPredictingEstimator(CalibratedClassifierCV(LogisticRegression(), method='isotonic'))
    model.fit(X, y)
    
    code = model_to_code(model, 'numpy', 'predict', 'test_model')#, substitutions=dict(zip(map(Symbol, ['x%d'%i for i in range(n)]), map(Symbol, X.columns))))
    numpy_test_module = exec_module('numpy_test_module', code)
    try:
        y_pred = numpy_test_module.test_model(**X)
        y_pred_correct = np.ravel(model.predict(X))
        assert_array_almost_equal(np.ravel(y_pred), y_pred_correct)
    except:
        print(code)
        print(X)
        raise
    
if __name__ == '__main__':
    test_sym_predict_calibrated_classifier_cv()
    exit()
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])


