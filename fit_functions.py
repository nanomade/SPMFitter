import numpy as np

def sine_fit_func(p, x):
    value = p[0] * np.sin(p[1] * x + p[2]) + p[3] + p[4] * x
    return value

def sine_2d_fit_func(p, x, y):
    value = p[0] * np.sin(p[1] * x + p[2]) + p[3] * np.sin(p[4] * y + p[5])
    return value

def sine_error_func(p, x, z):
    error = sine_fit_func(p, x) - z
    return error

def sine_2d_error_func(p, x, y, z):
    error = sine_2d_fit_func(p, x, y) - z
    # rms = np.sqrt(np.mean(error**2))
    # print('Optimizer! Error is: ', rms)
    return error


def top_hat(p, x, line):
    # p[0]: End of first low part of hat
    # p[1]: Start of second part of hat
    # p[2]: z-value of low part of 
    # p[3]: z-value of high part of 
    low = np.append(np.where(x < p[0])[0], np.where(x > p[1])[0])
    value = np.zeros(len(line)) + p[3]
    np.put(value, low, p[2])
    return value

def top_hat_error_func(p, x, y):
    error = top_hat(p, x, line=y) - y
    return error
