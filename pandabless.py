from sklearntools import STSimpleEstimator


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
        
    def fit(self, X, y, sample_weights=None, exposure=None):
        if self.predictors is None:
            self.predictors_ = list(X.columns)
        else:
            self.predictors_ = self.predictors
        if self.responses is None:
            self.responses_ = list(y.columns)
        else:
            self.responses_ = self.responses
        if self.sample_weights is None and sample_weights is not None:
            self.sample_weights_ = list(sample_weights.columns)
        else:
            self.sample_weights_ = self.sample_weights
        if self.exposure is None and exposure is not None:
            self.exposure_ = list(exposure.columns)
        else:
            self.exposure_ = self.exposure
    
        return self
    
    def _y_index(self):
        if self.responses_ is None:
            if self.responses is None:
                return None
            else:
                return self.responses
        else:
            return self.responses_
        
    def _x_index(self):
        if self.predictors_ is None:
            if self.predictors is None:
                return None
            else:
                return self.predictors
        else:
            return self.predictors_
    
    def _weight_index(self):
        if self.sample_weights_ is None:
            if self.sample_weights is None:
                return None
            else:
                return self.sample_weights
        else:
            return self.sample_weights_
    
    def _exposure_index(self):
        if self.exposure_ is None:
            if self.exposure is None:
                return None
            else:
                return self.exposure
        else:
            return self.exposure_
    
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
        