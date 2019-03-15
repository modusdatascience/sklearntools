from sklearntools.kfold import PredicateHybridCV,\
    column_interval_predicate
import numpy as np
from six.moves import reduce
from operator import __add__
from numpy.testing.utils import assert_array_equal
from nose.tools import assert_equal


def test_hybrid_cv():
    X = np.random.normal(size=(100,10))
    y = np.random.normal(size=100)
    cv = PredicateHybridCV(n_folds=10, predicate=column_interval_predicate(upper=1.))
#     ThresholdHybridCV(n_folds=10, upper=1.)
    folds = list(cv._iter_test_masks(X, y))
    assert_array_equal(reduce(__add__, folds), np.ones(100, dtype=int))
    assert_equal(len(folds), cv.get_n_splits(X, y))

if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])