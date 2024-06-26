import multiprocessing  # To be used for faster fitting

import gwyfile
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fit_functions

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


class SPMFitter:
    def __init__(self, filename):
        # Raw data as read from file
        self.original_data, self.size = self._load_file(filename)
        self.patterend_region = None
        self.modulated_region = None

        # List of data treatments that has been applied to the raw data
        self.treatments = []
        self.data = self.original_data

    def _load_file(self, filename):
        # TODO: Take other formats and convert with gwyddion command line
        obj = gwyfile.load(filename)
        channels = gwyfile.util.get_datafields(obj)
        # TODO: What about the other channels
        # TODO: Is main channel always 'ZSensor'?
        file_data = channels["ZSensor"].data
        x_size = channels["ZSensor"].xreal
        y_size = channels["ZSensor"].yreal
        return file_data, (x_size, y_size)

    # def find_all_medians(self):
    #     row_sum = np.zeros(len(self.data[1]))
    #     for i in range(0, len(self.data[1])):
    #         row_sum[i] = sum(self.data[:][i])
    #     fig, ax = plt.subplots()
    #     ax.plot(row_sum)
    #     plt.show()

    def _find_sub_area(self, area, mask=False):
        """
        Find a sub-area of self.data.
        :param tuple area: The area of relevance
        :param bool mask: If False, the result will be the sub-area indicated above
        If True, the result will be the full area with the area masked by nan
        :return: The sub-area or masked full area as
        """
        left_x = int(1e-6 * area[0][0] * self.data.shape[0] / self.size[0])
        right_x = int(1e-6 * area[1][0] * self.data.shape[0] / self.size[0])
        # A small poll in the office puts (0,0) in lower left corner
        # even though Gwiddion actually defaults to upper left corner
        low_y = int(
            self.data.shape[1] - (1e-6 * area[1][1] * self.data.shape[1] / self.size[1])
        )
        top_y = int(
            self.data.shape[1] - (1e-6 * area[0][1] * self.data.shape[1] / self.size[1])
        )
        # print('Lx: ', left_x, '  Rx: ', right_x, '  Ly: ', low_y, ' Ty: ', top_y)

        if mask:
            data = np.copy(self.data)
            data[low_y:top_y, left_x:right_x] = np.nan
        else:
            data = self.data[low_y:top_y, left_x:right_x]
        return data

    def calculate_roughness(self, area=None):
        # todo: RMS is just one way of calculating surface roughness
        # consider to implement other techniques
        if area is None:
            data = self.data
        else:
            data = self._find_sub_area(area)

        mean = data.sum() / data.size
        square_sum = np.square(data - mean).sum()
        square_mean = square_sum / data.size
        rms = np.sqrt(square_mean)
        return rms

    def _plane_fit(self, area=None, mask=False, plot=False):
        global_data = self.data
        if area is None:
            data = global_data
        else:
            data = self._find_sub_area(area, mask=mask)

        # https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
        X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        X_global, Y_global = np.meshgrid(
            np.arange(global_data.shape[1]), np.arange(global_data.shape[0])
        )

        G = np.ones((data.shape[0] * data.shape[1], 3))
        G[:, 0] = X.flatten()
        G[:, 1] = Y.flatten()
        Z = data.flatten()

        # Corresponds to:
        # for i in range(0, len(Z)):
        #     if np.isnan(Z[i]):
        #         delete_list.append(i)
        delete_list = np.argwhere(np.isnan(Z)).flatten()
        G = np.delete(G, delete_list, 0)
        Z = np.delete(Z, delete_list, 0)

        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
        # Begin magic, too long since I went to school....
        normal = (a, b, -1)
        nn = np.linalg.norm(normal)
        normal = normal / nn
        point = np.array([0.0, 0.0, c])
        d = -point.dot(normal)
        z = (-normal[0] * X_global - normal[1] * Y_global - d) * 1.0 / normal[2]
        # End of magic

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(X, Y, data, rstride=10, cstride=10)
            ax.plot_surface(X_global, Y_global, z, rstride=10, cstride=10, alpha=0.2)
            ax.plot_surface(
                X_global, Y_global, global_data - z, rstride=10, cstride=10, alpha=0.2
            )
            plt.show()
        return z

    def apply_plane_fit(self, area=None, mask=False, plot=True):
        fitted_plane = self._plane_fit(area, mask=mask, plot=plot)
        self.treatments.append('Subtract global plane fit')
        self.data = self.data - fitted_plane
        return True

    def _find_modulated_lines(self):
        """
        Find the modulated lines by comparing standard deviation of all lines,
        if it is higher than average, it is should be part of the modulation
        """
        stds = np.empty(len(self.data))
        for line_nr in range(0, len(self.data)):
            stds[line_nr] = self.data[line_nr][:][:].std()

        mean_std = stds.mean()
        modulated_lines = []
        for line_nr in range(0, len(self.data)):
            if stds[line_nr] > mean_std * 0.9:
                modulated_lines.append(line_nr)
        return modulated_lines

    def _index_to_area(self, x_l, x_r, y_t, y_b):
        low_left = (
            (1e6 * x_l * self.size[0] / self.data.shape[0]),
            (self.data.shape[1] - y_b) * 1e6 * self.size[1] / self.data.shape[1],
        )
        top_right = (
            (1e6 * x_r * self.size[0] / self.data.shape[0]),
            (self.data.shape[1] - y_t) * 1e6 * self.size[1] / self.data.shape[1],
        )
        return (low_left, top_right)

    def find_patterned_area(self, plot=False):
        patterned_lines = []
        left_edges = []
        right_edges = []
        # for line_nr in [88, 90, 112, 509]:
        # for line_nr in [100, 110, 506, 508, 510, 511]:
        for line_nr in range(0, len(self.data)):
            line = self.data[line_nr][:][:]
            X = np.arange(0, len(line))
            z = np.polyfit(X, line, 1)
            line = line - (X * z[0] + z[1])

            start_steps = []
            delta_ys = []
            for i in range(2, len(line)):
                delta_y = line[i] - line[i - 2]
                delta_ys.append(delta_y)
            hat_start = np.argmax(delta_ys)
            hat_stop = np.argmin(delta_ys)
            if (hat_stop - hat_start) < len(line) / 2:
                # This hat is obiously too small, not a patterned area
                continue

            low_part = np.append(line[:hat_start], line[hat_stop:])
            high_part = line[hat_start:hat_stop]
            p0 = [hat_start, hat_stop, np.mean(low_part), np.mean(high_part)]

            # Fit the hat as best as possible. Notice that the fit is unable to catch
            # the non-monotomic hat kink, this is hopefully correctly catched by
            # the initial guess if this is a patterned region
            fit = sp.optimize.least_squares(
                fit_functions.top_hat_error_func, p0[:], args=(X, line), **FIT_PARAMS
                #_top_hat_error_func, p0[:], args=(X, line), **FIT_PARAMS
            )

            # res = line - _top_hat(fit.x, X)
            # print(res.std())
            # print(fit.x)
            hat_amplitude = fit.x[3] - fit.x[2]
            line_amplitude = line.max() - line.min()

            if hat_amplitude / line_amplitude < 0.25:
                continue

            left_edges.append(hat_start)
            right_edges.append(hat_stop)
            patterned_lines.append(line_nr)

            # box_plot = []
            # for x in X:
            #    box_plot.append(_box(p0, x))

            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                # ax.plot(delta_ys, 'k+', label='Delta')
                ax.plot(line, 'r.', label='Data')
                # ax.plot(X*z[0]+z[1], 'g-', label='Data')
                # ax.plot(X, line, 'r+', label='Data')
                # ax.plot(X, res, 'b-', label='Init')
                ax.plot(X, _top_hat(p0, X), 'b-', label='Init')
                # ax.plot(X, _box(fit.x, X), 'g-', label='Fit')
                # ax.plot(X, box_plot, 'b-', label='Fit')
                # for peak in real_peaks:
                #    ax.plot(peak, line[peak], 'bo')
                # ax.hlines(1e-8, 0, 500)
                # ax.vlines(end_fit, line.min(), line.max())
                plt.show()

        print()

        left_edge = sorted(left_edges)[int(len(left_edges) * 0.2)]
        right_edge = sorted(right_edges)[int(len(right_edges) * 0.8)]
        low_left = (left_edge, patterned_lines[-1])
        top_right = (right_edge, patterned_lines[0])
        # return (low_left, top_right)
        area = self._index_to_area(
            left_edge, right_edge, patterned_lines[0], patterned_lines[-1]
        )
        return area

    def find_modulated_area(self, plot=False):
        modulated_lines = self._find_modulated_lines()
        center_line = int(len(modulated_lines) / 2)
        line = self.data[center_line][:][:]
        peaks, properties = sp.signal.find_peaks(line, distance=5, width=5)
        real_peaks = []
        for peak in peaks:
            if (line[peak] - line.mean()) > 0:
                real_peaks.append(peak)
        first_peak = real_peaks[0]
        last_peak = real_peaks[-1]
        period = real_peaks[1] - real_peaks[0]

        start_fit = first_peak - period / 4
        end_fit = last_peak + period / 4
        if plot:
            print('First peak: {}. Last peak: {}'.format(first_peak, last_peak))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            X = np.arange(0, len(line))
            ax.plot(X, line, 'r+', label='Data')
            for peak in real_peaks:
                ax.plot(peak, line[peak], 'bo')

            ax.vlines(start_fit, line.min(), line.max())
            ax.vlines(end_fit, line.min(), line.max())
            plt.show()

        low_left = (
            (1e6 * start_fit * self.size[0] / self.data.shape[0]),
            (self.data.shape[1] - modulated_lines[-1])
            * 1e6
            * self.size[1]
            / self.data.shape[1],
        )
        top_right = (
            (1e6 * end_fit * self.size[0] / self.data.shape[0]),
            (self.data.shape[1] - modulated_lines[0])
            * 1e6
            * self.size[1]
            / self.data.shape[1],
        )
        #
        # return (low_left, top_right)
        area = self._index_to_area(
            start_fit, end_fit, modulated_lines[0], modulated_lines[-1]
        )
        return area

    def fit_line(self, line, p0=None, pdfpage=None):
        dt = self.size[0] / len(self.original_data[0][:])
        X = np.arange(0, len(line))

        if p0 is None:
            peaks, _ = sp.signal.find_peaks(line, distance=20)
            # A decent guess for frequency is to distribute all peaks acress the line
            freq_guess = len(peaks) * 2 * np.pi / len(line)
            # We fit a sine, zero should be a quater period offset
            phase_guess = peaks[0] - (peaks[1] - peaks[0]) / 4
            # Amplitude guess is simply max minus min
            z_mean = sum(line) / len(line)
            ampl_guess = max(line) - z_mean
            p0 = [ampl_guess, freq_guess, phase_guess, z_mean, 0]

        fit = sp.optimize.least_squares(
            # self._sine_error_func, p0[:], args=(X, line), **FIT_PARAMS
            fit_functions.sine_error_func, p0[:], args=(X, line), **FIT_PARAMS
        )

        fit_params = {}
        if fit.status > 0:
            fit_params = {
                'amplitude': fit.x[0] * 1e9,  # nm
                'frequency': 1e-6 * fit.x[1] / dt,  # rad / μm
                'phase': fit.x[2] * dt * 1e9,  # nm
                'offset': fit.x[3] * 1e9,  # nm
                'slope': fit.x[4] * 1e3 / dt,  # nm/μm
            }

        if pdfpage:
            print(fit)
            fig = plt.figure()
            fig.subplots_adjust(right=0.75)
            ax = fig.add_subplot(2, 1, 1)
            ax.set_xlabel('Distance / μm', fontsize=6)
            ax.set_ylabel('Height / nm', fontsize=6)
            # for peak in peaks:
            #    ax.plot(peak, line[peak], 'bo')
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
            ax.tick_params(**TICK_PARMAMS)
            ax.yaxis.get_offset_text().set_size(6)
            ax.xaxis.get_offset_text().set_size(6)
            # plt.legend()

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
                ax.text(1.01, 1.0 - i * 0.07, msg, fontsize=6, transform=ax.transAxes)

            msg = 'Fit function:'
            ax.text(1.01, 0.6, msg, fontsize=6, transform=ax.transAxes)
            func = 'Asin(Fx + P) + O + Sx'
            ax.text(1.01, 0.54, func, fontsize=6, transform=ax.transAxes)

            msg = 'Optimality: {:.2e}'.format(fit.optimality)
            ax.text(1.01, 0.3, msg, fontsize=6, transform=ax.transAxes)
            msg = 'Iterations: {}'.format(fit.nfev)
            ax.text(1.01, 0.23, msg, fontsize=6, transform=ax.transAxes)
            msg = 'Success: {}.'.format(fit.success)
            ax.text(1.01, 0.16, msg, fontsize=6, transform=ax.transAxes)
            msg = 'Message:'
            ax.text(1.01, 0.09, msg, fontsize=6, transform=ax.transAxes)
            msg = '{}'.format(fit.message)
            ax.text(1.01, 0.02, msg, fontsize=6, transform=ax.transAxes)

            ax = fig.add_subplot(2, 1, 2)
            ax.set_xlabel('Distance / μm', fontsize=6)
            ax.set_ylabel('Residual / nm', fontsize=6)
            ax.plot(
                1e6 * dt * X,
                1e9 * (fit_functions.sine_fit_func(fit.x, X) - line),
                label='Residual',
            )
            ax.tick_params(**TICK_PARMAMS)
            ax.xaxis.get_offset_text().set_size(6)
            ax.yaxis.get_offset_text().set_size(6)

            plt.savefig(pdfpage, format='pdf')
            plt.close()
        return fit_params, fit

    def fit_to_all_lines(self, parameter, area=None, plot=False):
        # Set pp to None to skip exporting pdf
        pp = PdfPages('multipage.pdf')

        if area is None:
            data = self.data
        else:
            data = self._find_sub_area(area)

        fit = None
        values = []
        for line_nr in range(0, data.shape[0]):
            line = data[line_nr][:][:]

            # TODO: After gettings fit parameters from the first few fits, we could
            # gain a significant performance boost by funneling the rest of the
            # fits into a multi-process queue
            if fit:
                fit_params, fit = self.fit_line(line, p0=fit.x, pdfpage=pp)
            else:
                fit_params, fit = self.fit_line(line, pdfpage=pp)
            if fit is None:
                values.append(None)
            else:
                values.append(fit_params[parameter])
        if pp:
            pp.close()

        if plot:
            fig, ax = plt.subplots()
            ax.plot(values, 'r+', label=parameter)
            plt.show()
        return values

    def sinosodial_fit_area(self, area, plot=False):
        if area is None:
            print('You need to select an area')
            return
        data = self._find_sub_area(area)

        # https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
        X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        # We need a good guess for p0; fit_line actually is quite decent at guessing
        # ab-initio, so we simply pick a random line as starting guess for the 2d-fit
        _, line_fit = self.fit_line(data[2][:][:])
        p0 = line_fit.x
        fit = sp.optimize.least_squares(
            self._sine_error_func,
            p0[:],
            args=(X.flatten(), data.flatten()),
            **FIT_PARAMS
        )
        print(fit)

        init_data = np.ones((data.shape[0], data.shape[1]))
        for j in np.arange(0, data.shape[1]):
            init_data[:, j] = self._sine_fit_func(p0, j)

        fit_data = np.ones((data.shape[0], data.shape[1]))
        for j in range(0, data.shape[1]):
            fit_data[:, j] = self._sine_fit_func(fit.x, j)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            d = ax.imshow(data, interpolation='none', origin='upper')
            d.set_clim(data.min(), data.max())
            fig.colorbar(d)

            ax = fig.add_subplot(1, 3, 2)
            f = ax.imshow(fit_data, interpolation='none', origin='upper')
            f.set_clim(data.min(), data.max())
            fig.colorbar(f)

            ax = fig.add_subplot(1, 3, 3)
            r = ax.imshow(data - fit_data, interpolation='none', origin='upper')
            r.set_clim(data.min(), data.max())
            fig.colorbar(r)
        plt.show()


if __name__ == "__main__":
    # TODO:
    # - Fix y-axis on pdf export of fits to ensure shared y-axis
    # - FFT of residuals on line fits
    # - Estimate uncertainty on roughness
    # - Auto-detection of patterned and modulated areas

    FITTER = SPMFitter('F1.002.gwy')

    FITTER.apply_plane_fit(plot=False)

    print('Modulated: ', FITTER.find_modulated_area())
    print('Patterned: ', FITTER.find_patterned_area(plot=False))
    exit()

    area = (
        (1.5102501387263887, 0.6960142904897203),
        (10.333155927349708, 9.370975012535515),
    )
    FITTER.sinosodial_fit_area(area=area, plot=True)

    # todo:
    # FITTER.find_modulated_area()
    exit()

    # FITTER.fit_to_all_lines('frequency', plot=True)

    FITTER.apply_plane_fit()
    print(FITTER.calculate_roughness())

    # fit = FITTER.fit_line(7, plot=True)
    # fit = FITTER.fit_line(262, plot=True)

    Z = FITTER._plane_fit()
    FITTER.data = FITTER.data - Z
    # FITTER.plot_data()
    # FITTER.mask_patterned_area( (24, 120), (476, 476) )
    FITTER.find_all_medians()
