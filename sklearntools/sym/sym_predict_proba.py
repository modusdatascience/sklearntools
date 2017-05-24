from sklearn.linear_model.logistic import LogisticRegression
from .sym_predict import sym_predict_logistic_regression
from .base import call_method_or_dispatch
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from .adapters.gradient_boosting import sym_predict_proba_gradient_boosting_classifier

sym_predict_proba_dispatcher = {LogisticRegression: sym_predict_logistic_regression,
                                GradientBoostingClassifier: sym_predict_proba_gradient_boosting_classifier}
sym_predict_proba = call_method_or_dispatch('sym_predict_proba', sym_predict_proba_dispatcher)
