from sklearntools.concat import ConcatenatingEstimator
from sklearntools.calibration import IdentityTransformer
from sklearntools.super_learner import OrderTransformer, sort_rows_independently
import pandas
import numpy as np
from nose.tools import assert_list_equal
from numpy.testing.utils import assert_array_equal


def test_concatenating_estimator_transform():
    m = 100
    n = 10
    model = ConcatenatingEstimator([
                                    ('identity',IdentityTransformer()),
                                    ('order', OrderTransformer())
                                    ])
    data = pandas.DataFrame(np.random.normal(size=(m,n)), columns=['x%d' % i for i in range(n)])
    model.fit(data)
    output = model.transform(data)
    assert_list_equal(list(data.columns) + ['O(%d/%d)' % (i + 1, n) for i in range(n)], list(output.columns))
    assert_array_equal(output.iloc[:,:n], data)
    assert_array_equal(output.iloc[:,n:], 
                       sort_rows_independently(np.asarray(data), inplace=False))


if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])