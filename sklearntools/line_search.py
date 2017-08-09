from toolz.functoolz import curry
import math

@curry
def golden_section_search(tolerance, lower, upper, f, start, direction):
    a = lower
    b = upper
    phi = (1. + math.sqrt(5)) / 2.
    
    c = b - ((b-a)/phi)
    d = a + ((b-a)/phi)
    while abs(c-d) > tolerance:
        f_c = f(start + c*direction)
        f_d = f(start + d*direction)
        if f_c < f_d:
            b = d
#             f_b = f_d
        else:
            a = c
#             f_a = f_c
        
        c = b - ((b-a)/phi)
        d = a + ((b-a)/phi)
    return a + ((b - a) / 2.)

@curry
def zoom(first_step, max_steps, step_factor, f, start, direction):
    step_size = first_step
    f_0 = f(start)
    f_1 = f(start + step_size * direction)
    n_steps = 1
    while f_0 > f_1 and n_steps <= max_steps:
        step_size *= step_factor
        n_steps += 1
        f_1 = f(start + step_size * direction)
    return step_size

def zoom_search(searcher, zoomer, f, start, direction):
    return searcher(0., zoomer(f, start, direction), f, start, direction)


