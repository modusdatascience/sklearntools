from numpy import where, isnan, maximum, minimum, exp, logical_not, logical_and, logical_or, select, less_equal, greater_equal, less, nan
def ${function_name}(${', '.join(input_names)}):
%for line in assignment_code.splitlines():
    ${line}
%endfor
    ${return_code}