from .sklearntools import STSimpleEstimator, fit_predict, shrinkd
from sklearn.base import clone
from toolz.dicttoolz import valmap
from .line_search import golden_section_search, zoom_search, zoom
import numpy as np

class GradientBoostingEstimator(STSimpleEstimator):
    def __init__(self, base_estimator, loss_function, max_step_size=1., n_estimators=100):
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.max_step_size = max_step_size
        self.n_estimators = n_estimators
        
    def fit(self, X, y, sample_weight=None, exposure=None):
        
        initial_estimator = self.loss_function.init_estimator()
        fit_args = self._process_args(X=X, y=y, sample_weight=sample_weight, 
                                      exposure=exposure)
        y = fit_args.get('y')
        initial_estimator.fit(**fit_args)
        coefficients = [1.]
        estimators = [initial_estimator]
        predict_args = {'X':X}
        if exposure is not None:
            predict_args['exposure'] = exposure
        prediction = shrinkd(1, initial_estimator.predict(**valmap(shrinkd(1), predict_args)))
        gradient_args = {'y':y, 'pred':prediction}
        if sample_weight is not None:
            gradient_args['sample_weight': sample_weight]
        if exposure is not None:
            gradient_args['exposure': exposure]
        gradient = shrinkd(1, self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args)))
        partial_arguments = {'y':y}
        if sample_weight is not None:
            partial_arguments['sample_weight'] = sample_weight
        if exposure is not None:
            partial_arguments['exposure'] = exposure
        for _ in range(self.n_estimators):
            fit_args['y'] = gradient
            estimator = clone(self.base_estimator)
            approx_gradient = shrinkd(1, fit_predict(estimator, **valmap(shrinkd(1), fit_args)))
            loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
#             loss_grad = lambda pred: -self.loss_function.negative_gradient(pred=pred, **valmap(shrinkd(1), partial_arguments))
            alpha = zoom_search(golden_section_search(1e-12), zoom(1., 10, 2.), loss_function, approx_gradient, gradient)
#             alpha, _, _, _, _, _ = line_search(loss_function, loss_grad, shrinkd(1, prediction), shrinkd(1,gradient))
            if alpha is None:
#                 alpha = self.max_step_size / 10.
                raise ValueError('Line search did not converge')
            print 'alpha =', alpha
            estimators.append(estimator)
            coefficients.append(alpha)
            delta = alpha * approx_gradient
            prediction += delta
            if np.linalg.norm(delta) < 1e-10:
                break
            gradient_args['pred'] = prediction
            gradient = shrinkd(1, self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args)))
        self.coefficients_ = coefficients
        self.estimators_ = estimators
    
    def predict(self, X, exposure=None):
        predict_args = self._process_args(X=X,exposure=exposure)
        return sum(coef * est.predict(**predict_args) for est, coef in zip(self.estimators_, self.coefficients_))
        
            