from pyearth import Earth as PyEarth
from .sklearntools import STSimpleEstimator

class Earth(STSimpleEstimator, PyEarth):
    pass
