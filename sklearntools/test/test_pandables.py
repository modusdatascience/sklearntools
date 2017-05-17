from sklearntools.pandables import InputFixingTransformer
from sklearn.linear_model.base import LinearRegression
import pandas as pd
import numpy as np
import random

def test_input_fixing_transformer():
    m = 100
    n = 10
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=(n, 1))
    y = np.dot(X, beta)
    fit_X = pd.DataFrame(X, columns=['x%d' % i for i in range(n)])
    fit_y = pd.DataFrame(y, columns=['y'])
    score_X = pd.concat([fit_y, fit_X], axis=1)
    shuffled =list(fit_X.columns)
    random.shuffle(shuffled)
    predict_X = fit_X[shuffled]
    
    model = InputFixingTransformer() >> LinearRegression()
    model.fit(fit_X, fit_y)
    score = model.score(score_X)
    pred = model.predict(predict_X)
    
    np.testing.assert_array_almost_equal(y, pred)
    np.testing.assert_almost_equal(score, 1.)
    
if __name__ == '__main__':
    test_input_fixing_transformer()
    print 'Success!'
    



