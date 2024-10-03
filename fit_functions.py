import numpy as np

def sine_fit_func(p, x):
    value = p[0] * np.sin(p[1] * x + p[2]) + p[3] + p[4] * x
    return value

def sine_error_func(p, x, y):
    error = sine_fit_func(p, x) - y
    return error

