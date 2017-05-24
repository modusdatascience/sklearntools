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

%for function_code in functions:
${function_code}
%endfor
