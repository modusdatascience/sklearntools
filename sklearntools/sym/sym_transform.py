from pyearth.earth import Earth
from pyearth.export import export_sympy_term_expressions
from .base import call_method_or_dispatch

sym_transform_dispatcher = {Earth: export_sympy_term_expressions}
sym_transform = call_method_or_dispatch('sym_transform', sym_transform_dispatcher)
