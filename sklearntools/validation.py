import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.metrics.ranking import roc_curve, auc

def plot_roc(observed, predicted, name, iterations=100, **kwargs):
    fpr, tpr, thresholds = roc_curve(observed, predicted)
    roc_auc = auc(fpr, tpr)
    bsauc = []
    for _ in range(iterations):
        obs, pred = resample(observed, predicted)
        fpr_, tpr_, thresholds = roc_curve(obs, pred)
        bsauc.append(auc(fpr_, tpr_))
    bsauc = np.asarray(bsauc)
    lower = np.percentile(bsauc, 2.5)
    upper = np.percentile(bsauc, 97.5)
    plt.plot(fpr, tpr, label='Model %s (AUC = %0.3f, 95%% CI %0.3f-%0.3f)' % (name, roc_auc, lower, upper), **kwargs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc=0)

class MovingWindowStatistic(object):
    def __init__(self, window_size, stat):
        self.window_size = window_size
        self.stat = stat
        
    def __call__(self, x, order):
        x_sorted = np.ravel(x)[order]
        result = []
        for start in range(0, len(x_sorted) - self.window_size):
            end = start + self.window_size
            data = x_sorted[start:end]
            result.append(self.stat(data))
        return np.array(result)

class BinStatistic(object):
    def __init__(self, n_bins, stat):
        self.n_bins = n_bins
        self.stat = stat
    
    def __call__(self, x, order):
        x_sorted = np.ravel(x)[order]
        remainder = x_sorted.shape[0] % self.n_bins
        quotient = x_sorted.shape[0] // self.n_bins
        
        result = []
        start = 0
        while start < x_sorted.shape[0]:
            size = quotient
            if remainder > 0:
                size += 1
                remainder -= 1
            end = start + size
            data = x_sorted[start:end]
            result.append(self.stat(data))
            start = end
        
#         if remainder > self.window_size / 2:
#             first_last 
#         result = []
#         for start in range(0, len(x_sorted) - self.window_size):
#             end = start + self.window_size
#             data = x_sorted[start:end]
#             result.append(self.stat(data))
        return np.array(result)

class CategoryStatistic(object):
    def __init__(self, stat):
        self.stat = stat
        
    def __call__(self, x, category_table):
        result = []
        for col in category_table:
            cat_dat = np.asarray(x)[np.asarray(category_table[col])]
            if len(cat_dat) > 0:
                result.append(self.stat(cat_dat))
            else:
                result.append(float('nan'))
        return np.asarray(result)

# class CategoryStatistic(object):
#     def __init__(self, stat):
#         self.stat = stat
#         
#     def __call__(self, x, category_index, categories):
#         result = []
#         for category in categories:
#             result.append(self.stat(x[category_index == category]))
#         return np.array(result)

def percentile(p):
    return lambda x: np.nanpercentile(x, p)

def quantile(q):
    return percentile(100. * q)

mean = np.mean

std = np.std

def mean_pm_std(factor):
    return lambda x: mean(x) + factor * std(x)
 
def bootstrap(outer_stat, inner_stat, n):
    def _bootstrap(*args):
        result = np.zeros(shape=n)
        for i in range(n):
            inner_args = resample(*[np.asarray(arg) for arg in args])
            if type(inner_args) is not list:
                inner_args = [inner_args]
            result[i] = inner_stat(*inner_args)
        return outer_stat(result)
    return _bootstrap

def statistic_error_bar_plot(StatClass):
    def _statistic_error_bar_plot(x, y, n, x_statistic, center_statistic, lower_statistic, upper_statistic, 
                        center_kwargs, *args, **kwargs):
        xstat = StatClass(n, x_statistic)
        centerstat = StatClass(n, center_statistic)
        lowerstat = StatClass(n, lower_statistic)
        upperstat = StatClass(n, upper_statistic)
        
        order = np.argsort(np.ravel(x))
        
        x_plot = xstat(x, order)
        y_center = centerstat(y, order)
        y_lower = y_center - lowerstat(y, order)
        y_upper = upperstat(y, order) - y_center
        
        y_err = np.concatenate([y_lower[:, None], y_upper[:, None]], axis=1)
        if center_kwargs is None:
            center_kwargs = {}
        plt.plot(x_plot, y_center, **center_kwargs)
        plt.errorbar(x_plot, y_center, yerr=y_err.T, *args, **kwargs)
    
    return _statistic_error_bar_plot

moving_window_error_bar_plot = statistic_error_bar_plot(MovingWindowStatistic)
bin_error_bar_plot = statistic_error_bar_plot(BinStatistic)

def category_error_bar_plot(category_table, y, center_statistic, lower_statistic, upper_statistic, center_kwargs,
                            *args, **kwargs):
    centerstat = CategoryStatistic(center_statistic)
    lowerstat = CategoryStatistic(lower_statistic)
    upperstat = CategoryStatistic(upper_statistic)
    
    category_names = list(category_table.columns)
    x_plot = np.arange(len(category_names))
    y_center = centerstat(y, category_table)
    y_lower = y_center - lowerstat(y, category_table)
    y_upper = upperstat(y, category_table) - y_center

    y_err = np.concatenate([y_lower[:, None], y_upper[:, None]], axis=1)
    
    if center_kwargs is None:
        center_kwargs = {}
    plt.plot(x_plot, y_center, **center_kwargs)
    
    plt.errorbar(x_plot, y_center, yerr=y_err.T, *args, **kwargs)
#     plt.xticks(x_plot, category_names)


# def category_error_bar_plot(category_array, category_names, y, center_statistic, lower_statistic, upper_statistic, 
#                         center_kwargs, *args, **kwargs):
#     centerstat = CategoryStatistic(center_statistic)
#     lowerstat = CategoryStatistic(lower_statistic)
#     upperstat = CategoryStatistic(upper_statistic)
#     
#     x_plot = np.arange(len(category_names))
#     y_center = centerstat(y, category_array, category_names)
#     y_lower = y_center - lowerstat(y, category_array, category_names)
#     y_upper = upperstat(y, category_array, category_names) - y_center
#     
#     y_err = np.concatenate([y_lower[:, None], y_upper[:, None]], axis=1)
#     
#     if center_kwargs is None:
#         center_kwargs = {}
#     plt.plot(x_plot, y_center, **center_kwargs)
#     plt.errorbar(x_plot, y_center, yerr=y_err.T, *args, **kwargs)
#     plt.xlabel(category_names)

def statistic_plot(StatClass):
    def _statistic_plot(x, y, n, x_statistic, y_statistic, *args, **kwargs):
        xstat = StatClass(n, x_statistic)
        ystat = StatClass(n, y_statistic)
        
        order = np.argsort(np.ravel(x))
        
        x_plot = xstat(x, order)
        y_plot = ystat(y, order)
        
        plt.plot(x_plot, y_plot, *args, **kwargs)
    return _statistic_plot

moving_window_plot = statistic_plot(MovingWindowStatistic)
bin_plot = statistic_plot(BinStatistic)
def category_plot(category_table, y, statistic, *args, **kwargs):
    cols = list(category_table.columns)
    x_plot = np.arange(len(cols))
    y_plot = CategoryStatistic(statistic)(y, category_table)
    plt.plot(x_plot, y_plot, *args, **kwargs)
    plt.xticks(x_plot, cols)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.xlim(x_plot[0] - .05 * x_plot[-1], x_plot[-1] + .05 * x_plot[-1])
    
# def category_plot(category_array, category_names, y, statistic, *args, **kwargs):
#     stat = CategoryStatistic(statistic)
#     
#     x_plot = np.arange(len(category_names))
#     y_plot = stat(y, category_array, category_names)
#     
#     plt.plot(x_plot, y_plot, *args, **kwargs)
#     plt.xlabel(category_names)
#     


def calibration_category_plot(category_table, observed, predicted,  obs_stat=mean, pred_stat=mean, pred_function=lambda obs, pred: pred, 
                              percent=95, n_bootstraps=100, observed_label='Observed', predicted_label='Predicted', title=None):
    error_bar_kwargs = {'marker':'.', 'linestyle':''}
    if observed_label is not None:
        error_bar_kwargs['label'] = observed_label
     
    category_error_bar_plot(category_table, pred_function(observed, predicted), obs_stat, bootstrap(quantile(.5 - percent / 200.), obs_stat, n_bootstraps), 
                            bootstrap(quantile(.5 + percent / 200.), obs_stat, n_bootstraps), error_bar_kwargs, linestyle='')
     
    if predicted_label is not None:
        category_plot(category_table, pred_function(observed, predicted), pred_stat, '.', label=predicted_label)
    plt.legend(loc='best')
    if title is not None:
        plt.title(title)

def calibration_bin_plot(covariate, observed, predicted, obs_stat=mean, pred_stat=mean, pred_function=lambda obs, pred: pred, 
                         percent=95, n_bins=20, n_bootstraps=100, observed_label='Observed', predicted_label='Predicted', title=None):
    error_bar_kwargs = {'marker':'+', 'linestyle':''}
    if observed_label is not None:
        error_bar_kwargs['label'] = observed_label
    bin_error_bar_plot(covariate, observed, n_bins, quantile(.5), obs_stat, bootstrap(quantile(.5 - percent / 200.), obs_stat, n_bootstraps), 
                       bootstrap(quantile(.5 + percent / 200.), obs_stat, n_bootstraps), error_bar_kwargs, linestyle='')
#     bin_plot(y_pred, y, n_bins, quantile(.5), mean, '.', label='Mean Observation')
#     bin_plot(y_pred, y, n_bins, quantile(.5), mean_pm_std(1.), '.', label='Mean Observation + STD')
#     bin_plot(y_pred, y, n_bins, quantile(.5), mean_pm_std(-1.), '.', label='Mean Observation - STD')
    if predicted_label is not None:
        bin_plot(covariate, predicted, n_bins, quantile(.5), pred_stat, '.', label=predicted_label)
    plt.legend(loc='best')
    if title is not None:
        plt.title(title)

# def background_plot(x, y, window_size, statistics, alphas, labels, *args, **kwargs):
#     

def quantile_background_plot(x, y, window_size, quantiles=[.025,.25,.5,.75,.975], *args, **kwargs):
    quantiles = np.ravel(np.array(quantiles))
    alphas = 1. - np.maximum(quantiles, 1-quantiles)
    
    order = np.argsort(np.ravel(x))
    xstat = MovingWindowStatistic(window_size, quantile(.5))
    x_plot = xstat(x, order)
    for q, a in zip(quantiles, alphas):
        ystat = MovingWindowStatistic(window_size, quantile(q))
        y_plot = ystat(y, order)
        
        kw = kwargs.copy()
        if 'alpha' not in kw:
            kw['alpha'] = a
        else:
            kw['alpha'] = a * kw['alpha']
        plt.plot(x_plot, y_plot, *args, **kw)

# def std_background_plot(x, y, window_size, factor, *args, **kwargs):
    
    
    
    
    
    

def _plot_moving_windows():
    m = 10000
    w = 200
    p05 = MovingWindowStatistic(w, percentile(5.))
    p95 = MovingWindowStatistic(w, percentile(95.))
    mn = MovingWindowStatistic(w, mean)
    sd = MovingWindowStatistic(w, std)
    
    x = np.random.uniform(0,100,size=m)
    y = np.random.normal(10. * np.sin(x / 10.), 1 + 3 * np.cos(x / 12.) ** 2)
    order = np.argsort(x)
    
    x_mn = mn(x, order)
    y_mn = mn(y, order)
    y_p05 = p05(y, order)
    y_p95 = p95(y, order)
    y_sd = sd(y, order)
    
    plt.plot(x_mn, y_p95, label='95%')
    plt.plot(x_mn, y_mn + y_sd, label='+sd')
    plt.plot(x_mn, y_mn, label='mean')
    plt.plot(x_mn, y_mn - y_sd, label='-sd')
    plt.plot(x_mn, y_p05, label='5%')
    plt.legend(loc='best')
    plt.show()
    
if __name__ == '__main__':
    _plot_moving_windows()
    
    
    

        

















