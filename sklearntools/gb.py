from .sklearntools import fit_predict, shrinkd, LinearCombination, BaseDelegatingEstimator, \
    STSimpleEstimator, AlreadyFittedEstimator, growd
from sklearn.base import clone
from toolz.dicttoolz import valmap, dissoc
from .line_search import golden_section_search, zoom_search, zoom
import numpy as np
from sklearn.ensemble.gradient_boosting import RegressionLossFunction,\
    QuantileEstimator, QuantileLossFunction, ExponentialLoss,\
    ScaledLogOddsEstimator
from operator import __sub__, __lt__
from toolz.itertoolz import sliding_window
from itertools import starmap
from toolz.functoolz import flip, curry
from sklearn.exceptions import NotFittedError
from .sym.sym_predict import sym_predict
from .sym.sym_score_to_decision import sym_score_to_decision
from .sym.syms import syms
from .sym.sym_score_to_proba import sym_score_to_proba
from distutils.version import LooseVersion
import sklearn
from types import MethodType
from sklearn.linear_model.base import LinearRegression
from scipy.optimize.linesearch import line_search
import traceback

# Patch over bug in scikit learn (issue #9539)
if LooseVersion(sklearn.__version__) <= LooseVersion('0.18.2'):
    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        diff = y - pred
        alpha = self.alpha
    
        mask = y > pred
        if sample_weight is None:
            loss = (alpha * diff[mask].sum() -
                    (1.0 - alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            loss = ((alpha * np.sum(sample_weight[mask] * diff[mask]) -
                    (1.0 - alpha) * np.sum(sample_weight[~mask] * diff[~mask])) /
                    sample_weight.sum())
        return loss
    QuantileLossFunction.__call__ = MethodType(__call__, None, QuantileLossFunction)


if LooseVersion(sklearn.__version__) <= LooseVersion('0.19.1'):
#     def __call__(self, y, pred, sample_weight=None):
#         pred = pred.ravel()
#         if sample_weight is None:
#             return np.mean(np.exp(- y * pred))
#         else:
#             return (1.0 / sample_weight.sum() *
#                     np.sum(sample_weight * np.exp(- y * pred)))
    def negative_gradient(self, y, pred, **kargs):
        y_ = -(2. * y - 1.)
        return - y_ * np.exp(y_ * pred.ravel())
    
    ExponentialLoss.negative_gradient = MethodType(negative_gradient, None, ExponentialLoss)



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

class STExponentialLossFunction(object):
    def init_estimator(self):
        return ScaledLogOddsEstimator()
    
    def __call__(self, y, pred, sample_weight=None):
        print y
        print pred
        pred = pred.ravel()
        if sample_weight is None:
            return np.exp(- y * pred)
        else:
            return np.sum(sample_weight * np.exp(- y * pred))
    
    def negative_gradient(self, y, pred, sample_weight=None, **kargs):
        result = - y * np.exp(- y * pred.ravel())
        if sample_weight is not None:
            return sample_weight * result
        return result
#         y_ = -(2. * y - 1.)
#         return y_ * np.exp(y_ * pred.ravel()) 
    
class SmoothQuantileLossFunction(RegressionLossFunction):
    def __init__(self, n_classes, tau, alpha):
        super(SmoothQuantileLossFunction, self).__init__(n_classes)
        self.tau = tau
        self.alpha = alpha
    
    def init_estimator(self):
        return QuantileEstimator(self.tau)
    
    def __call__(self, y, pred, sample_weight=None):
        x = y - pred
#         n = float(y.shape[0])
        if sample_weight is not None:
            return np.dot(sample_weight, (self.tau * x + self.alpha * log_one_plus_exp_x(-(1./self.alpha)*x)))
        else:
            return np.sum(self.tau * x + self.alpha * log_one_plus_exp_x(-(1./self.alpha)*x))
    
    def negative_gradient(self, y, pred, sample_weight=None):
        x =  y - pred
#         n = float(y.shape[0])
        if sample_weight is not None:
            return  sample_weight * (self.tau - one_over_one_plus_exp_x((1. / self.alpha) * x))
        else:
            return (self.tau - one_over_one_plus_exp_x((1. / self.alpha) * x))
        
    def _update_terminal_region(self, *args, **kwargs):
        raise NotImplementedError()

def never_stop_early(**kwargs):
    return False

class NIterationsWithoutImprovementOverThreshold(object):
    def __init__(self, stat, n, threshold=0.):
        self.stat = stat
        self.n = n
        self.threshold = threshold
    
    def __call__(self, losses, **kwargs):
        if len(losses) <= self.n:
            return False
        return all(map(curry(__lt__)(-self.threshold), starmap(self.stat, sliding_window(2, losses[-(self.n+1):]))))
@curry
def stop_after_n_iterations_without_stat_improvement_over_threshold(stat, n, threshold=0.):
    return NIterationsWithoutImprovementOverThreshold(stat, n, threshold)
#     def _stop_after_n_iterations_without_improvement(losses, **kwargs):
#         if len(losses) <= n:
#             return False
#         return all(map(curry(__lt__)(-threshold), starmap(stat, sliding_window(2, losses[-(n+1):]))))
#     return _stop_after_n_iterations_without_improvement

def reduction(before, after):
    return after - before

stop_after_n_iterations_without_improvement_over_threshold = stop_after_n_iterations_without_stat_improvement_over_threshold(flip(__sub__))

def percent_reduction(before, after):
    return 100*(after - before) / float(before)

stop_after_n_iterations_without_percent_improvement_over_threshold = stop_after_n_iterations_without_stat_improvement_over_threshold(percent_reduction)

# class GradientDescentRegressor(STSimpleEstimator):
#     def __init__(self, loss_function, initial_value=0.):
#         self.loss_function = loss_function
#     
#     def fit(self, X, y, sample_weight=None):
#         intercept = self.initial_value
#         coef = np.zeros(X.shape[1])
#         prediction = intercept + np.dot(X, coef)
#         gradient_args = {'y':y, 'pred':prediction}
#         while True:
#             gradient = self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args))
#             linear_model = LinearRegression().fit(X, gradient)
#             approx_gradient = linear_model.predict(X)
#             
        

class GradientBoostingEstimator(BaseDelegatingEstimator):
    def __init__(self, base_estimator, loss_function, learning_rate=.1, n_estimators=100,
                 stopper=never_stop_early, verbose=0, extra_fit=False):
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.stopper = stopper
        self.verbose = verbose
        self.extra_fit = extra_fit
        
    def fit(self, X, y, sample_weight=None, exposure=None, previous_prediction=None):
        
        fit_args = {'X': growd(2,X), 'y': shrinkd(1,y)}
        if sample_weight is not None:
            fit_args['sample_weight'] = shrinkd(1, sample_weight)
        if exposure is not None:
            fit_args['exposure'] = shrinkd(1, exposure)
#         self._process_args(X=X, y=y, sample_weight=sample_weight, 
#                                       exposure=exposure)
        y = growd(2,y)
        coefficients = []
        estimators = []
        if previous_prediction is None:
            initial_estimator = self.loss_function.init_estimator()
            initial_estimator.fit(**fit_args)
            coefficients.append(1.)
            estimators.append(initial_estimator)
        predict_args = {'X':X}
        if exposure is not None:
            predict_args['exposure'] = exposure
        if previous_prediction is None:
            prediction = shrinkd(1, initial_estimator.predict(**valmap(shrinkd(1), predict_args)))
        else:
            prediction = previous_prediction.copy()
            
#         prediction_cv = prediction.copy()
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
        loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
        self.initial_loss_ = loss_function(prediction)
        loss = self.initial_loss_
#         loss_cv = loss
        losses = [self.initial_loss_]
#         losses_cv = [self.initial_loss_]
        predict_args = {'X': X}
        if exposure is not None:
            predict_args['exposure'] = shrinkd(1, exposure)
        self.early_stop_ = False
        for iteration in range(self.n_estimators):
            previous_loss = loss
#             previous_loss_cv = loss_cv
            if self.verbose >= 1:
                print('Fitting estimator %d...' % (iteration + 1))
            fit_args['y'] = shrinkd(1, gradient)
            estimator = clone(self.base_estimator)
            try:
                approx_gradient = shrinkd(1, fit_predict(estimator, **fit_args))
            except:
                raise
            if self.verbose >= 1:
                print('Fitting for estimator %d complete.' % (iteration + 1))
#             approx_gradient = estimator.predict(**predict_args)
#             prediction_cv += alpha * approx_gradient_cv
#             loss_cv = loss_function(prediction_cv)
            
            if self.extra_fit:
                transform_args = {'X': X}
                if exposure is not None:
                    transform_args['exposure'] = exposure 
                extra_fit_args = {'X': estimator.transform(**transform_args), 'y': y, 'previous_prediction': prediction}
                if sample_weight is not None:
                    extra_fit_args['sample_weight'] = sample_weight
                extra_predict_args = dissoc(extra_fit_args, 'y', 'previous_prediction')
                extra_estimator = GradientBoostingEstimator(LinearRegression(), loss_function=self.loss_function,
                                                            stopper=stop_after_n_iterations_without_percent_improvement_over_threshold(1, .001))
                extra_estimator.fit(**valmap(shrinkd(1), extra_fit_args))
                prediction += extra_estimator.decision_function(**extra_predict_args)
                loss = loss_function(prediction)
                coefficients.append(1.)
                estimators.append((AlreadyFittedEstimator(estimator) >> AlreadyFittedEstimator(extra_estimator)).fit(X,y))
                losses.append(loss)
            else:
#                 loss_grad = lambda pred: -self.loss_function.negative_gradient(pred=pred, **valmap(shrinkd(1), partial_arguments))
                if self.verbose >= 1:
                    print('Computing alpha for estimator %d...' % (iteration + 1))
#                 alpha, _, _, _, _, _ = line_search(loss_function, loss_grad, shrinkd(1, prediction), shrinkd(1,gradient))
                alpha = zoom_search(golden_section_search(1e-16), zoom(1., 20, 2.), loss_function, prediction, approx_gradient)
                alpha *= self.learning_rate
                if self.verbose >= 1:
                    print('alpha = %f' % alpha)
                if self.verbose >= 1:
                    print('Computing alpha for estimator %d complete.' % (iteration + 1))
            
                prediction += alpha * approx_gradient
                loss = loss_function(prediction)
                coefficients.append(alpha)
                estimators.append(estimator)
                losses.append(loss)
#                 losses_cv.append(loss_cv)
            if self.verbose >= 1:
                print('Loss after %d iterations is %f, a reduction of %f%%.' % (iteration + 1, loss, 100*(previous_loss - loss)/float(previous_loss)))
#                 if loss_cv != loss:
#                     print('Cross-validated loss after %d iterations is %f, a reduction of %f%%.' % (iteration + 1, loss_cv, 100*(previous_loss_cv - loss_cv)/float(previous_loss_cv)))
                print('Checking early stopping condition for estimator %d...' % (iteration + 1))
            
            if self.stopper(iteration=iteration, coefficients=coefficients, losses=losses, 
                            gradient=gradient, approx_gradient=approx_gradient):#, approx_gradient_cv=approx_gradient_cv):
                self.early_stop_ = True
                if self.verbose >= 1:
                    print('Stopping early after %d iterations.' % (iteration + 1))
                break
            if self.verbose >= 1:
                print('Not stopping early.')
            gradient_args['pred'] = prediction
            gradient = shrinkd(1, self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args)))
        self.coefficients_ = coefficients
        self.estimators_ = estimators
        self.losses_ = losses
#         self.losses_cv_ = losses_cv
        self.score_ = (self.initial_loss_ - loss) / self.initial_loss_
        self.estimator_ = LinearCombination(self.estimators_, self.coefficients_)
        self._create_delegates('estimator', ['syms'])
        return self
    
    def statistic_over_steps(self, X, y, statistic, exposure=None):
        result = []
        predict_args = {'X': X}
        if exposure is not None:
            predict_args['exposure'] = exposure
        for j in range(1, len(self.estimators_)+1):
            model = LinearCombination(self.estimators_[:j], self.coefficients_[:j])
            pred = model.predict(X)
            result.append(statistic(y, pred))
        return result

    def score(self, X, y, sample_weight=None, exposure=None):
        partial_arguments = self._process_args(y=y, sample_weight=sample_weight, exposure=exposure)
        predict_arguments = self._process_args(X=X, exposure=exposure)
        loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
        prediction = shrinkd(1, self.predict(**predict_arguments))
        loss = loss_function(prediction)
        initial_prediction = shrinkd(1, self.coefficients_[0] * self.estimators_[0].predict(**predict_arguments))
        initial_loss = loss_function(initial_prediction)
        return (initial_loss - loss) / initial_loss
    
    def transform(self, X, exposure=None):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        args = self._process_args(X=X, exposure=exposure)
        return np.concatenate([est.transform(**args) for est in self.estimators_[1:]], axis=1)
    
    def predict(self, X, exposure=None):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        pred_args = self._process_args(X=X, exposure=exposure)
        score = self.estimator_.predict(**pred_args)
        if hasattr(self.loss_function, '_score_to_decision'):
            return self.loss_function._score_to_decision(score)
        else:
            return score
    
    def sym_predict(self):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        inner = sym_predict(self.estimator_)
        if hasattr(self.loss_function, '_score_to_decision'):
            outer = sym_score_to_decision(self.loss_function)
            variable = syms(self.loss_function)[0]
            return outer.subs({variable: inner})
        else:
            return inner
        
    def predict_proba(self, X, exposure=None):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        if hasattr(self.loss_function, '_score_to_proba'):
            pred_args = self._process_args(X=X, exposure=exposure)
            score = self.estimator_.predict(**pred_args)
            return self.loss_function._score_to_proba(score)
        else:
            raise AttributeError()
    
    def sym_predict_proba(self):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        inner = sym_predict(self.estimator_)
        if hasattr(self.loss_function, '_score_to_proba'):
            outer = sym_score_to_proba(self.loss_function)
            variable = syms(self.loss_function)[0]
            return outer.subs({variable: inner})
        else:
            return inner
    
    def decision_function(self, X, exposure=None):
        if not hasattr(self.loss_function, '_score_to_decision'):
            raise AttributeError()
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        pred_args = self._process_args(X=X, exposure=exposure)
        score = self.estimator_.predict(**pred_args)
        return score
    
    def sym_decision_function(self):
        if not hasattr(self.loss_function, '_score_to_decision'):
            raise AttributeError()
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        return sym_predict(self.estimator_)
        
    
    @property
    def _estimator_type(self):
        if hasattr(self.loss_function, '_score_to_decision'):
            return 'classifier'
        else:
            return 'regressor'
        