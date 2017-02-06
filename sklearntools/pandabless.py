from .sklearntools import STSimpleEstimator
from sympy.core.symbol import Symbol
from sym import syms_x, syms


class Pandable(object):
    pass

class InputFixingTransformer(STSimpleEstimator, Pandable):
    '''
    When fit, memorizes the columns of the input dataframes.  Later, grabs the appropriate columns 
    from inputs during updates and transforms.
    '''
    
    def __init__(self, predictors=None, responses=None, sample_weights=None, exposure=None):
        self.predictors = predictors
        self.responses = responses
        self.sample_weights = sample_weights
        self.exposure = exposure
    
    def syms(self):
        try:
            return [Symbol(predictor) for predictor in self.predictors_]
        except TypeError:
            return syms_x(self)
        
    def sym_transform(self):
        return syms(self)
        
    def input_size(self):
        return len(self.predictors_)
        
    def fit(self, X, y, sample_weights=None, exposure=None):
        if self.predictors is None:
            self.predictors_ = list(X.columns)
#         else:
#             self.predictors_ = [self.predictors]
        if self.responses is None:
            self.responses_ = list(y.columns)
#         else:
#             self.responses_ = self.responses
        if self.sample_weights is None and sample_weights is not None:
            self.sample_weights_ = list(sample_weights.columns)
#         else:
#             self.sample_weights_ = self.sample_weights
        if self.exposure is None and exposure is not None:
            self.exposure_ = list(exposure.columns)
#         else:
#             self.exposure_ = self.exposure
    
        return self
    
    def _y_index(self):
        if hasattr(self, 'responses_') and self.responses_ is None:
            if self.responses is None:
                return None
            else:
                return self.responses
        elif hasattr(self, 'responses_'):
            return self.responses_
        else:
            return None
        
    def _x_index(self):
        if hasattr(self, 'predictors_') and self.predictors_ is None:
            if self.predictors is None:
                return None
            else:
                return self.predictors
        elif hasattr(self, 'predictors_'):
            return self.predictors_
        else:
            return None
    
    def _weight_index(self):
        if hasattr(self, 'sample_weights_') and self.sample_weights_ is None:
            if self.sample_weights is None:
                return None
            else:
                return self.sample_weights
        elif hasattr(self, 'sample_weights_'):
            return self.sample_weights_
        else:
            return None
    
    def _exposure_index(self):
        if hasattr(self, 'exposure_') and self.exposure_ is None:
            if self.exposure is None:
                return None
            else:
                return self.exposure
        elif hasattr(self, 'exposure_'):
            return self.exposure_
        else:
            return None
    
    def transform(self, X, exposure=None):
        x_index = self._x_index()
        return X[x_index]
        
    def update(self, data):
        y_index = self._y_index()
        x_index = self._x_index()
        weight_index = self._weight_index()
        exposure_index = self._exposure_index()
        
        if 'y' in data:
            if y_index is not None:
                data['y'] = data['y'][y_index]
        elif y_index is not None:
            try:
                data['y'] = data['X'][y_index]
            except KeyError:
                pass
        
        if 'sample_weight' in data:
            if weight_index is not None:
                data['sample_weight'] = data['sample_weight'][weight_index]
        elif weight_index is not None:
            try:
                data['sample_weight'] = data['X'][weight_index]
            except KeyError:
                pass
        
        if 'exposure' in data:
            if exposure_index is not None:
                data['exposure'] = data['exposure'][exposure_index]
        elif exposure_index is not None:
            try:
                data['exposure'] = data['X'][exposure_index]
            except KeyError:
                pass
        
        if 'X' in data:
            if x_index is not None:
                data['X'] = data['X'][x_index]
        