from sklearntools.validation import plot_tolerance, plot_roc,\
    calibration_bin_plot, plot_roc_auc_for_bins




if __name__ == '__main__':
    from sklearntools.validation import plot_curve_auc, roc_curve
    from sklearn.linear_model.logistic import LogisticRegression
    from scipy.special._ufuncs import expit
    import numpy as np
    from matplotlib import pyplot
    np.random.seed(1)
    m = 1000
    n = 5
    
    X = np.random.normal(size=(m,n))
    beta = np.random.normal(size=n)
    y = np.random.binomial(n=1, p=expit(np.dot(X, beta)))
    
    model = LogisticRegression().fit(X, y)
    pred = model.predict_proba(X)[:,1]
    
    pyplot.figure()
    plot_roc(y, pred, name='test_model')
    pyplot.savefig('test_roc_plot.png')
    
    pyplot.figure()
    plot_tolerance(y, pred, name='test_model', normalize=True)
    pyplot.savefig('test_tolerance_plot.png')
    
    pyplot.figure()
    calibration_bin_plot(pred, y, pred, )
    pyplot.savefig('test_calibration_plot.png')
    
    pyplot.figure()
    plot_roc_auc_for_bins(10, X[:,0], y, pred)
    pyplot.savefig('test_bin_auc_plot.png')