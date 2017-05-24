from sklearn.linear_model.logistic import LogisticRegression
from .sym_predict import sym_predict_logistic_regression
from .base import call_method_or_dispatch

sym_predict_proba_dispatcher = {LogisticRegression: sym_predict_logistic_regression}
sym_predict_proba = call_method_or_dispatch('sym_predict_proba', sym_predict_proba_dispatcher)
