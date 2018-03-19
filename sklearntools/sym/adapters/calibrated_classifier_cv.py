from sklearn.exceptions import NotFittedError
from ..input_size import input_size, register_input_size
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier
from ..sym_predict_proba import register_sym_predict_proba,\
    sym_predict_proba
from six.moves import reduce
from operator import __add__
from sympy.core.numbers import RealNumber, Zero
from ..sym_predict import sym_predict
from ..syms import syms, register_syms
from ..base import fallback
from ..sym_decision_function import sym_decision_function

# @register_input_size(CalibratedClassifierCV)
# def input_size_calibrated_classifier_cv(estimator):
#     if not hasattr(estimator, 'calibrated_classifiers_'):
#         raise NotFittedError()
#     return input_size(estimator.calibrated_classifiers_[0])

@register_sym_predict_proba(CalibratedClassifierCV)
def sym_predict_proba_calibrated_classifier_cv(estimator):
    if not hasattr(estimator, 'calibrated_classifiers_'):
        raise NotFittedError()
    return reduce(__add__, map(sym_predict_proba, estimator.calibrated_classifiers_)) / RealNumber(len(estimator.calibrated_classifiers_))

@register_syms(CalibratedClassifierCV)
def syms_calibrated_classifier_cv(estimator):
    return syms(estimator.calibrated_classifiers_[0])

# @register_input_size(_CalibratedClassifier)
# def input_size__calibrated_classifier(estimator):
#     return input_size(estimator.base_estimator)

@register_syms(_CalibratedClassifier)
def syms__calibrated_classifier(estimator):
    return syms(estimator.base_estimator)

@register_sym_predict_proba(_CalibratedClassifier)
def sym_predict_proba__calibrated_classifier(estimator):
    if hasattr(estimator.base_estimator, 'decision_function'):
        inner_pred = sym_decision_function(estimator.base_estimator)
    elif hasattr(estimator.base_estimator, 'predict_proba'):
        inner_pred = sym_predict_proba(estimator.base_estimator)
#     inner_pred = fallback(sym_decision_function, sym_predict_proba)(estimator.base_estimator)
    result = Zero()
    for cal in estimator.calibrators_:
        variables = syms(cal)
        if len(variables) != 1:
            raise ValueError()
        var = variables[0]
        result += sym_predict(cal).subs({var: inner_pred})
    return result / RealNumber(len(estimator.calibrators_))

def sym_predict_proba_parts__calibrated_classifier(estimator):
    if hasattr(estimator.base_estimator, 'decision_function'):
        inner_pred = sym_decision_function(estimator.base_estimator)
    elif hasattr(estimator.base_estimator, 'predict_proba'):
        inner_pred = sym_predict_proba(estimator.base_estimator)
    result = Zero()
    var = None
    for cal in estimator.calibrators_:
        variables = syms(cal)
        if len(variables) != 1 or (var != variables[0] and var is not None):
            raise ValueError()
        var = variables[0]
        result += sym_predict(cal)
    result = result / RealNumber(len(estimator.calibrators_))
    return ((var,), [result], (syms(estimator.base_estimato), inner_pred, None))
