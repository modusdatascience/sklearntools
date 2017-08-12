from .sklearntools import STSimpleEstimator, fit_predict, shrinkd
from sklearn.base import clone
from toolz.dicttoolz import valmap
from .line_search import golden_section_search, zoom_search, zoom
import numpy as np
from sklearn.ensemble.gradient_boosting import RegressionLossFunction,\
    QuantileEstimator, QuantileLossFunction
from scipy.optimize.linesearch import line_search
from matplotlib import pyplot as plt

def log_one_plus_exp_x(x):
    lower = -10.
    upper = 35.
    result = np.zeros_like(x)
    low_idx = x < lower
    result[low_idx] = np.exp(x[low_idx])
    high_idx = x > upper
    result[high_idx] = x[high_idx]
    middle_idx = ~(low_idx | high_idx)
    result[middle_idx] = np.log(1+np.exp(x[middle_idx]))
    return result

def one_over_one_plus_exp_x(x):
    lower = -100.
    upper = 100.
    result = np.zeros_like(x)
    low_idx = x < lower
    result[low_idx] = 1.
    high_idx = x > upper
    result[high_idx] = 0.
    middle_idx = ~(low_idx | high_idx)
    result[middle_idx] = 1. / (1. + np.exp(x[middle_idx]))
    return result

class SmoothQuantileLossFunction(RegressionLossFunction):
    def __init__(self, n_classes, tau, alpha):
        super(SmoothQuantileLossFunction, self).__init__(n_classes)
        self.tau = tau
        self.alpha = alpha
    
    def init_estimator(self):
        return QuantileEstimator(self.tau)
    
    def __call__(self, y, pred, sample_weight=None):
        x = y - pred
        n = float(y.shape[0])
        if sample_weight is not None:
            return (1/n) * np.dot(sample_weight, (self.tau * x + self.alpha * log_one_plus_exp_x(-(1./self.alpha)*x)))
        else:
            return (1/n) * np.sum(self.tau * x + self.alpha * log_one_plus_exp_x(-(1./self.alpha)*x))
    
    def negative_gradient(self, y, pred, sample_weight=None):
        x =  y - pred
        n = float(y.shape[0])
        if sample_weight is not None:
            return  sample_weight * (self.tau - one_over_one_plus_exp_x((1. / self.alpha) * x))
        else:
            return (self.tau - one_over_one_plus_exp_x((1. / self.alpha) * x))
        
    def _update_terminal_region(self, *args, **kwargs):
        raise NotImplementedError()
    
class GradientBoostingEstimator(STSimpleEstimator):
    def __init__(self, base_estimator, loss_function, max_step_size=1., n_estimators=100):
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.max_step_size = max_step_size
        self.n_estimators = n_estimators
        
    def fit(self, X, y, sample_weight=None, exposure=None):
        qloss = QuantileLossFunction(1, self.loss_function.tau)
        qqloss = SmoothQuantileLossFunction(1, self.loss_function.tau, 0.000001)
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
        y_base = y.copy()
        if sample_weight is not None:
            partial_arguments['sample_weight'] = sample_weight
        if exposure is not None:
            partial_arguments['exposure'] = exposure
        loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
#         loss_grad = lambda pred: -self.loss_function.negative_gradient(pred=pred, **valmap(shrinkd(1), partial_arguments))
        self.initial_loss_ = loss_function(prediction)
        loss = self.initial_loss_
        losses = [self.initial_loss_]
        for i in range(self.n_estimators):
            print 'loss =', loss, self.loss_function(np.ravel(y),np.ravel(prediction))
            print 'qloss =', qloss(np.ravel(y),np.ravel(prediction))
            print 'qqloss =', qqloss(np.ravel(y),np.ravel(prediction))
            assert np.all(partial_arguments['y'] == y_base)
            fit_args['y'] = gradient
            print gradient
            estimator = clone(self.base_estimator)
            approx_gradient = shrinkd(1, fit_predict(estimator, **valmap(shrinkd(1), fit_args)))
            print 'norm(gradient) =', np.linalg.norm(gradient)
            print 'norm(approx_gradient) =', np.linalg.norm(approx_gradient)
            print 'gradient . approx_gradient =', np.dot(gradient, approx_gradient)
#             alpha = zoom_search(golden_section_search(1e-12), zoom(1., 20, 2.), loss_function, prediction, approx_gradient)
#             alpha, _, _, _, _, _ = line_search(loss_function, loss_grad, shrinkd(1, prediction), shrinkd(1,approx_gradient))
            alpha = zoom_search(golden_section_search(1e-12), zoom(1., 20, 2.), loss_function, prediction, approx_gradient)
            print 'alpha =', alpha
#             print 'other_alpha =', other_alpha
#             x = np.arange(-50, 50, .1)
#             y = np.empty_like(x)
#             for i in range(len(x)):
#                 y[i] = loss_function(prediction + approx_gradient * x[i])
#             plt.plot(x, y)
#             plt.show()
#             z = np.empty_like(x)
#             for i in range(len(x)):
#                 z[i] = loss_function(prediction + gradient * x[i])
#             plt.plot(x, y)
#             plt.show()
            if alpha is None:
#                 alpha = self.max_step_size / 10.
#                 alpha = zoom_search(golden_section_search(1e-12), zoom(1., 20, 2.), loss_function, prediction, approx_gradient)
                raise ValueError('Line search did not converge')
            
            estimators.append(estimator)
            coefficients.append(alpha)
            delta = alpha * approx_gradient
            prediction += delta
            loss = loss_function(prediction)
            
            losses.append(loss)
#             if np.linalg.norm(delta) / float(len(delta)) < 1e-10:
#                 pass
            if abs(losses[-1] - losses[-2]) < 1e-10:
                print 'early terminate after %d iterations' % (i+1)
                break
            print 'loss_diff =', abs(losses[-1] - losses[-2])
            gradient_args['pred'] = prediction
            gradient = shrinkd(1, self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args)))
            
#             if np.linalg.norm(gradient) / float(len(gradient)) < 1e-10:
#                 break
        print 'all %d iterations completed' % (i+1)
        self.coefficients_ = coefficients
        self.estimators_ = estimators
        self.losses_ = losses
        self.score_ = (self.initial_loss_ - loss) / self.initial_loss_
    
    def score(self, X, y, sample_weight=None, exposure=None):
        partial_arguments = self._process_args(y=y, sample_weight=sample_weight, exposure=exposure)
        predict_arguments = self._process_args(X=X, exposure=exposure)
        loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
        prediction = shrinkd(1, self.predict(**predict_arguments))
        loss = loss_function(prediction)
        initial_prediction = shrinkd(1, self.coefficients_[0] * self.estimators_[0].predict(**predict_arguments))
        initial_loss = loss_function(initial_prediction)
        return (initial_loss - loss) / initial_loss
        
    def predict(self, X, exposure=None):
        predict_args = self._process_args(X=X,exposure=exposure)
        return sum(shrinkd(1, coef * est.predict(**valmap(shrinkd(1),predict_args))) for est, coef in zip(self.estimators_, self.coefficients_))
        
            