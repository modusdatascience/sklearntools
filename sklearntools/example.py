'''
Created on Feb 15, 2016

@author: jason
'''

from .sklearntools import MultipleResponseEstimator, BackwardEliminationEstimatorCV, \
    QuantileRegressor, ResponseTransformingEstimator
from pyearth import Earth
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

outcomes = ['admission_rate', 'prescription_cost_rate', '']

[('earth', Earth(max_degree=2)), ('elim', BackwardEliminationEstimatorCV())]


