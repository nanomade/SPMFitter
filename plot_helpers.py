import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import fit_functions
import constants


def _format_ax(ax, x_label, y_label):
    ax.tick_params(**constants.TICK_PARMAMS)
    ax.legend(fontsize=4)
    ax.set_xlabel(x_label, fontsize=5)
    ax.set_ylabel(y_label, fontsize=5)
    ax.yaxis.get_offset_text().set_size(5)
    ax.xaxis.get_offset_text().set_size(5)


# REMOVE XF!!!!!!!!!!!!!!!!!!1
def write_line_fit_pdf_page(pdfpage, line, fit, dt, p0, fft, fit_params):
    X = np.arange(0, len(line))

    print(fit)
    fig = plt.figure()
    fig.subplots_adjust(right=0.95)

    # MAIN DATA PLOT
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(1e6 * dt * X, 1e9 * line, 'r+', label='Data')
    ax.plot(
        1e6 * dt * X,
        1e9 * fit_functions.sine_fit_func(p0, X),
        linewidth=0.2,
        label='Initial function',
    )
    ax.plot(
        1e6 * dt * X,
        1e9 * fit_functions.sine_fit_func(fit.x, X),
        label='Fitted function',
    )
    _format_ax(ax, 'Distance / μm', 'Height / nm')

    texts = [
        ('Amplitude (A): {:.2f}nm', 'amplitude'),
        ('Frequency (F): {:.2f}rad/μm', 'frequency'),
        ('Phase (P): {:.1f}nm', 'phase'),
        ('Offset (O): {:.1f}nm', 'offset'),
        ('Slope (S): {:.1f} nm/μm', 'slope'),
    ]
    for i in range(0, len(texts)):
        key = texts[i][1]
        msg = texts[i][0].format(fit_params.get(key, -1))
        ax.text(1.1, 1.0 - i * 0.07, msg, fontsize=6, transform=ax.transAxes)

    msg = 'Fit function:'
    ax.text(1.1, 0.6, msg, fontsize=6, transform=ax.transAxes)
    func = 'Asin(Fx + P) + O + Sx'
    ax.text(1.1, 0.54, func, fontsize=6, transform=ax.transAxes)

    msg = 'Optimality: {:.2e}'.format(fit.optimality)
    ax.text(1.1, 0.3, msg, fontsize=6, transform=ax.transAxes)
    msg = 'Iterations: {}'.format(fit.nfev)
    ax.text(1.1, 0.23, msg, fontsize=6, transform=ax.transAxes)
    msg = 'Success: {}.'.format(fit.success)
    ax.text(1.1, 0.16, msg, fontsize=6, transform=ax.transAxes)
    msg = 'Message:'
    ax.text(1.1, 0.09, msg, fontsize=6, transform=ax.transAxes)
    msg = '{}'.format(fit.message)
    ax.text(1.1, 0.02, msg, fontsize=6, transform=ax.transAxes)

    # (2, 2, 2)  does not exist, it is used for the text

    # RESIDUAL PLOT
    residual = fit_functions.sine_fit_func(fit.x, X) - line
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(
        1e6 * dt * X,
        1e9 * residual,
        label='Residual',
    )
    _format_ax(ax, 'Distance / μm', 'Height / nm')

    # FFT PLOT
    ax = fig.add_subplot(2, 2, 4)
    residual_fft = abs(sp.fft.fft(residual))
    # This x-axis is shared
    xf = sp.fft.fftfreq(len(line), dt)[: len(residual_fft) // 2]
    ax.plot(xf, fft[: len(fft) // 2], 'r.', label='Before fit')
    ax.plot(xf, residual_fft[: len(residual_fft) // 2], 'b.', label='After fit')

    ax.ylim = (0, 10)

    _format_ax(ax, 'Freqeuncy / μm$^-$$^1$', 'Amplitude / nm')

    plt.savefig(pdfpage, format='pdf')
    plt.close()
