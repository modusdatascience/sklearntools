from .base import call_method_or_dispatch, fallback
from .sym_transform import sym_transform

sym_update_dispatcher = {}
sym_update = fallback(call_method_or_dispatch('sym_update', sym_update_dispatcher), sym_transform)
