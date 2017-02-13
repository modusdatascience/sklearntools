import numpy as np
from earth import Earth
from sklearn.linear_model.logistic import LogisticRegression
from calibration import ProbaPredictingEstimator,\
    ThresholdClassifier
import pandas
from sym import javascript_str, model_to_code, python_str
from numpy.ma.testutils import assert_array_almost_equal
import imp
import execjs

def exec_module(name, code):
    module = imp.new_module(name)
    exec code in module.__dict__
    return module

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
    y = np.random.binomial(1, 1. / (1. + np.exp(-eta)))
    
    model = Earth(allow_missing=True, max_terms=10) >> Earth(allow_missing=True, max_terms=10) >> ProbaPredictingEstimator(ThresholdClassifier(LogisticRegression()))
    model.fit(X, y)
    
    print model_to_code(model, 'numpy', 'predict', 'test_model')
    numpy_test_module = exec_module('numpy_test_module', model_to_code(model, 'numpy', 'predict', 'test_model'))
    y_pred = numpy_test_module.test_model(col3=X['col3'], col8=X['col8'])
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))
    
    python_test_module = exec_module('python_test_module', model_to_code(model, 'python', 'predict', 'test_model'))
    y_pred = [python_test_module.test_model(col3=row['col3'], col8=row['col8']) for i, row in X.iterrows()]
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))
    
    js = execjs.get()
    context = js.compile(model_to_code(model, 'javascript', 'predict', 'test_model'))
    y_pred = [context.eval('test_model(col3=%s, col8=%s)' % (str(row['col3']) if not np.isnan(row['col3']) else 'NaN', 
                                                             str(row['col8']) if not np.isnan(row['col8']) else 'NaN')) 
              for i, row in X.iloc[:10,:].iterrows()]
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X.iloc[:10,:])))
    
if __name__ == '__main__':
    test_sympy_export()
    print 'Success!'
    
    



