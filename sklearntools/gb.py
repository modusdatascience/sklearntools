from .sklearntools import STSimpleEstimator, fit_predict
from sklearn.base import clone
from scipy.optimize.linesearch import line_search
from toolz.curried import partial
import numpy as np


class GradientBoostingEstimator(STSimpleEstimator):
    def __init__(self, base_estimator, loss_function, max_step_size=1., n_estimators=100):
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.max_step_size = max_step_size
        
    def fit(self, X, y, sample_weight=None, exposure=None):
        
        initial_estimator = self.loss_function.init_estimator()
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight, 
                                      exposure=exposure)
        initial_estimator.fit(**fit_args)
        coefficients = [1.]
        estimators = [initial_estimator]
        predict_args = {'X':X}
        if exposure is not None:
            predict_args['exposure'] = exposure
        prediction = initial_estimator.predict(**predict_args)
        gradient_args = {'y':y, 'pred':prediction}
        if sample_weight is not None:
            gradient_args['sample_weight': sample_weight]
        if exposure is not None:
            gradient_args['exposure': exposure]
        gradient = self.loss_function(**gradient_args)
        partial_arguments = {'y':y}
        if sample_weight is not None:
            partial_arguments['sample_weight'] = sample_weight
        if exposure is not None:
            partial_arguments['exposure'] = exposure
        for _ in range(self.n_estimators):
            fit_args['y'] = gradient
            estimator = clone(self.base_estimator)
            approx_gradient = fit_predict(estimator, **fit_args)
            alpha, _ = line_search(partial(self.loss_function, **partial_arguments),
                                partial(self.loss_function.negative_gradient, **partial_arguments),
                                prediction, approx_gradient, amax=self.max_step_size)
            if alpha is None:
                raise ValueError('Line search did not converge')
            estimators.append(estimator)
            coefficients.append(alpha)
            prediction += alpha * approx_gradient
        self.coefficients_ = coefficients
        self.estimators_ = estimators
    
    def predict(self, X, exposure=None):
        predict_args = self.process_args(X=X,exposure=exposure)
        return np.dot([est.predict(**predict_args) for est in self.estimators_], self.coefficients_)
        
            