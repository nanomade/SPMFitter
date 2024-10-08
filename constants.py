TICK_PARMAMS = {  # This should go to a file with defaults
    'direction': 'in',
    'length': 3,
    'width': 1,
    'colors': 'k',
    'labelsize': 5,
    'axis': 'both',
    'pad': 2,
}

FIT_PARAMS = {
    'method': 'lm',
    'jac': '2-point',
    'ftol': 1e-14,
    'xtol': 1e-14,
    'max_nfev': 20000,
}
