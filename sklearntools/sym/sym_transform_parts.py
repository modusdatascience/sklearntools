from .base import call_method_or_dispatch, fallback
from .parts import double_check
from .syms import syms
from .sym_transform import sym_transform

def sym_transform_parts_base(obj, target=None):
    return (syms(obj), sym_transform(obj), target)

sym_transform_parts_dispatcher = {}
sym_transform_parts = double_check(fallback(call_method_or_dispatch('sym_transform_parts', sym_transform_parts_dispatcher), sym_transform_parts_base))
