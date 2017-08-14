from sympy.core.numbers import RealNumber
from sympy.functions.elementary.exponential import exp
from sklearn.ensemble.gradient_boosting import BinomialDeviance,\
    LogOddsEstimator, GradientBoostingClassifier, QuantileEstimator,\
    LossFunction
from ..base import call_method_or_dispatch
from operator import add
from ..sym_predict_proba import register_sym_predict_proba
from ..sym_predict import sym_predict
from sklearntools.sym.input_size import register_input_size,\
    input_size_from_n_features
from sklearntools.sym.sym_predict import register_sym_predict
from sympy.core.symbol import Symbol
from sklearntools.sym.syms import register_syms
from sympy import exp
from sklearntools.sym.sym_score_to_proba import register_sym_score_to_proba,\
    sym_score_to_proba
from sympy.functions.elementary.miscellaneous import Max
from sklearntools.sym.sym_score_to_decision import register_sym_score_to_decision
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import true

def sym_binomial_deviance_score_to_proba(loss, score):
    return RealNumber(1) / (RealNumber(1) + exp(RealNumber(-1) * score))

sym_loss_function_dispatcher = {BinomialDeviance: sym_binomial_deviance_score_to_proba}
sym_loss_function = call_method_or_dispatch('sym_loss_function', sym_loss_function_dispatcher)


def sym_log_odds_estimator_predict(estimator):
    return RealNumber(estimator.prior)

sym_init_function_dispatcher = {LogOddsEstimator: sym_log_odds_estimator_predict}
sym_init_function = call_method_or_dispatch('sym_init_function', sym_init_function_dispatcher)

def sym_predict_proba_gradient_boosting_classifier(estimator):
    learning_rate = RealNumber(estimator.learning_rate)
    trees = map(sym_predict, estimator.estimators_[:,0])
    tree_part = learning_rate * reduce(add, trees)
    init_part = sym_init_function(estimator.init_)
    return sym_loss_function(estimator.loss_, tree_part + init_part)

register_sym_predict_proba(GradientBoostingClassifier, sym_predict_proba_gradient_boosting_classifier)
register_input_size(GradientBoostingClassifier, input_size_from_n_features)

def sym_predict_quantile_estimator(estimator):
    return RealNumber(estimator.quantile)

register_sym_predict(QuantileEstimator, sym_predict_quantile_estimator)

@register_sym_predict(LogOddsEstimator)
def sym_predict_log_odds_estimator(estimator):
    return RealNumber(estimator.prior)

@register_syms(LossFunction)
def syms_loss_function(loss):
    return [Symbol('x')]

@register_sym_score_to_proba(BinomialDeviance)
def sym_score_to_proba_binomial_deviance(loss):
    return 1 / (1 + exp(-Symbol('x')))

@register_sym_score_to_decision(BinomialDeviance)
def sym_score_to_decision(loss):
    return Piecewise((RealNumber(1), sym_score_to_proba(loss) > RealNumber(1)/RealNumber(2)), (RealNumber(0), true))



