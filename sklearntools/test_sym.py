import numpy as np
from earth import Earth
from sklearn.linear_model.logistic import LogisticRegression
from calibration import ProbaPredictingEstimator,\
    ThresholdClassifier
from sym import STNumpyPrinter, STJavaScriptPrinter
import pandas
from sym import javascript_str, numpy_str, python_str
from numpy.ma.testutils import assert_array_almost_equal

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
    
    model = Earth(allow_missing=True, max_terms=10) >> ProbaPredictingEstimator(ThresholdClassifier(LogisticRegression()))
    model.fit(X, y)
    
    expression = model.sym_predict()
    print model.intermediate_stages_[0].summary()
    print expression
    printer = STNumpyPrinter()
    print printer.doprint(expression)
    jsprinter = STJavaScriptPrinter()
    print jsprinter.doprint(expression)
    print javascript_str('test_model', model)
    print numpy_str('test_model', model)
    print python_str('test_model', model)
    
    import sys,imp
    python_code = numpy_str('test_model', model)
    test_module = imp.new_module('test_module')
    exec python_code in test_module.__dict__
    
    y_pred = test_module.test_model(col3=X['col3'], col8=X['col8'])
    assert_array_almost_equal(np.ravel(y_pred), np.ravel(model.predict(X)))
    
if __name__ == '__main__':
    test_sympy_export()

    
    



