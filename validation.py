import numpy as np

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

def percentile(p):
    return lambda x: np.nanpercentile(x, p)

def quantile(q):
    return percentile(100. * q)

mean = np.mean

std = np.std

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
    
    from matplotlib import pyplot as plt
    plt.plot(x_mn, y_p95, label='95%')
    plt.plot(x_mn, y_mn + y_sd, label='+sd')
    plt.plot(x_mn, y_mn, label='mean')
    plt.plot(x_mn, y_mn - y_sd, label='-sd')
    plt.plot(x_mn, y_p05, label='5%')
    plt.legend(loc='best')
    plt.show()
    
if __name__ == '__main__':
    _plot_moving_windows()
    
    
    

        

















