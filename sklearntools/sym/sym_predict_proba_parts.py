from .syms import syms
from .sym_predict_proba import sym_predict_proba
from .base import fallback, call_method_or_dispatch
from .parts import double_check
from sklearntools.sym.base import create_registerer

def sym_predict_proba_parts_base(obj, target=None):
    return (syms(obj), [sym_predict_proba(obj)], target)

sym_predict_proba_parts_dispatcher = {}
sym_predict_proba_parts = double_check(fallback(call_method_or_dispatch('sym_predict_proba_parts', sym_predict_proba_parts_dispatcher), sym_predict_proba_parts_base))
register_sym_predict_proba_parts = create_registerer(sym_predict_proba_parts_dispatcher, 'register_sym_predict_proba_parts')

