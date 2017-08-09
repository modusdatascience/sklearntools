from sklearntools.line_search import golden_section_search, zoom, zoom_search
from nose.tools import assert_almost_equal
import numpy as np

def test_golden_section_search():
    f = lambda x: (x-2) ** 2
    alpha = golden_section_search(1e-12, 0, 10, f, 0., 1.)
    assert_almost_equal(alpha, 2.)
    
    alpha = golden_section_search(1e-12, 0, 10, f, -1., 1.)
    assert_almost_equal(alpha, 3.)

    alpha = golden_section_search(1e-12, 0, 10, f, -1., 2.)
    assert_almost_equal(alpha, 3./2.)
    
    f = lambda x: (x[0]-1)**2 + (x[1]-2)**2
    alpha = golden_section_search(1e-12, 0, 10, f, np.array([0.,0.]), np.array([2., 4.]))
    assert_almost_equal(alpha, .5)

def test_zoom():
    f = lambda x: (x-2) ** 2
    alpha = golden_section_search(1e-12, 0, zoom(1., 10, 2., f, 0., 1.), f, 0., 1.)
    assert_almost_equal(alpha, 2.)
    
    alpha = zoom_search(golden_section_search(1e-12), zoom(1., 10, 2.), f, 0., 1.)
    assert_almost_equal(alpha, 2.)
    
    f = lambda x: -x
    alpha = zoom_search(golden_section_search(1e-12), zoom(1., 10, 2.), f, 0., 1.)
    assert_almost_equal(alpha, 2.**10)


if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])