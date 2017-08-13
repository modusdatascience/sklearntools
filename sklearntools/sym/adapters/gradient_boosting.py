from sympy.core.numbers import RealNumber
from sympy.functions.elementary.exponential import exp
from sklearn.ensemble.gradient_boosting import BinomialDeviance,\
    LogOddsEstimator, GradientBoostingClassifier, QuantileEstimator
from ..base import call_method_or_dispatch
from operator import add
from ..sym_predict_proba import register_sym_predict_proba
from ..sym_predict import sym_predict
from sklearntools.sym.input_size import register_input_size,\
    input_size_from_n_features
from sklearntools.sym.sym_predict import register_sym_predict

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
