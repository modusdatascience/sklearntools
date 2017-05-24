from sympy.core.numbers import RealNumber
from sympy.functions.elementary.exponential import exp
from sklearn.ensemble.gradient_boosting import BinomialDeviance,\
    LogOddsEstimator
from ..base import call_method_or_dispatch
from .trees import sym_predict_decision_tree_regressor
from operator import add

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
    trees = map(sym_predict_decision_tree_regressor, estimator.estimators_[:,0])
    tree_part = learning_rate * reduce(add, trees)
    init_part = sym_init_function(estimator.init_)
    return sym_loss_function(estimator.loss_, tree_part + init_part)
