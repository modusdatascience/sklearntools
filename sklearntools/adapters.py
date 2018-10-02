from multipledispatch.dispatcher import Dispatcher
from xgboost.sklearn import XGBRegressor

predict = Dispatcher(name='predict')

@predict.register(object)
def predict_default(estimator, *args, **kwargs):
    return estimator.predict(*args, **kwargs)

@predict.register(XGBRegressor)
def predict_xgb_regressor(estimator, X, **kwargs):
    return estimator.predict(X, **kwargs)
