function missing(val) {
  if (val !== val) {
    return 1;
  } else {
    return 0;
  }
};

function nanprotect(val) {
  if (val !== val) {
    return 0;
  } else {
    return val;
  }
};

function ${function_name}(${', '.join(input_names)}) {
%for line in assignment_code.splitlines():
    ${line};
%endfor
    ${return_code};
};
