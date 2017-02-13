from math import isnan, exp
def nanprotect(val):
    if isnan(val):
        return 0.
    else:
        return val

def missing(val):
    if isnan(val):
        return 1.
    else:
        return 0.

def negate(val):
    if val:
        return 0.
    else:
        return 1.

def ${function_name}(${', '.join(input_names)}):
%for line in assignment_code.splitlines():
    ${line}
%endfor
    ${return_code}
