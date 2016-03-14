from sklearn.metrics import log_loss


def log_loss_scorer(clf, X, y, **kwargs):
    y_pred = clf.predict(X, **kwargs)
    args = {'y_true':y>0, 'y_pred':y_pred}
    if 'sample_weight' in kwargs:
        args['sample_weight'] = kwargs['sample_weight']
    
    return log_loss(**args)




