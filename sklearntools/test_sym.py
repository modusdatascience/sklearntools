import numpy as np
from earth import Earth
from sklearn.linear_model.logistic import LogisticRegression
from calibration import ProbaPredictingEstimator,\
    ThresholdClassifier
from sym import STNumpyPrinter, STJavaScriptPrinter
import pandas
from sym import javascript_str, syms

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
    
    
if __name__ == '__main__':
    test_sympy_export()

    
    



