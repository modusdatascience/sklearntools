import numpy as np
from sklearntools.interpolation import LinearInterpolation, LinearInterpolationArray
from nose.tools import assert_equal
from numpy.testing.utils import assert_approx_equal, assert_array_almost_equal

def test_bisect_left():
    
    x = np.arange(0., 1., .1)
    y = x ** 2
    lower = 0.
    upper = 1.
    
    interp = LinearInterpolation(x, y, lower, lower, upper, upper)
    assert_equal(interp.bisect_left(-1.), -1)
    assert_equal(interp.bisect_left(.51), 5)
    assert_equal(interp.bisect_left(1.), 9)

def test_call():
    x = np.arange(0., 1., .1)
    y = x ** 2
    lower = 0.
    upper = 1.
    
    interp = LinearInterpolation(x, y, lower, lower, upper, upper)
    assert_approx_equal(interp(.55), ((.5 ** 2) + (.6 ** 2)) / 2.)
    assert_approx_equal(interp(0.), 0.)
    assert_approx_equal(interp(1.), 1.)
    
def test_linear_interpolation_array_empty_get_set_item():
    x = np.arange(0., 1., .1)
    y = x ** 2
    lower = 0.
    upper = 1.
    
    interp = LinearInterpolation(x, y, lower, lower, upper, upper)
    arr = LinearInterpolationArray.empty(10)
    arr[:] = interp
    assert_equal(arr[1:3], LinearInterpolationArray(np.array([interp]*2)))

def test_linear_interpolation_array_call():
    x = np.arange(0., 1., .0001)
    y = x ** 2
    lower = 0.
    upper = 1.
    
    n = 100000
    interp = LinearInterpolation(x, y, lower, lower, upper, upper)
    arr = LinearInterpolationArray.empty(n)
    arr[:] = interp
    assert_array_almost_equal(arr(.5), .25)
    
    v = np.random.uniform(size=n)
    assert_array_almost_equal(arr(v), v ** 2)
    
    
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])