from pyearth.earth import Earth
from memorize import memorize
from sklearn.externals.joblib.memory import Memory
import os
from sklearn.datasets.samples_generator import make_regression
from shutil import rmtree

def test_memorization():
    memory_dir = 'test_memory_dir'
    if os.path.exists(memory_dir):
        rmtree(memory_dir)
    try:
        model = memorize(Memory(memory_dir), Earth())
        X, y = make_regression()
        model.fit(X, y)
        assert not model.loaded_from_cache_
        model2 = memorize(Memory(memory_dir), Earth())
        model2.fit(X, y)
        assert model2.loaded_from_cache_
    finally:
        if os.path.exists(memory_dir):
            rmtree(memory_dir)
    
if __name__ == '__main__':
    import sys
    import nose
    # This code will run the test in this file.'
    module_name = sys.modules[__name__].__file__

    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s', '-v'])
