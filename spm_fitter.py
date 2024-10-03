import multiprocessing  # To be used for faster fitting

import gwyfile
import ruptures as rpt 
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

        # TODO!!!! THIS SHOULD BE DONE AUTOMATICALLY!!!!!!!!!!!
        # self.original_data = sp.ndimage.rotate(self.original_data, 1.3)  # AFM
        self.original_data = sp.ndimage.rotate(self.original_data, 0)  # NF

        self.patterend_region = None  # Todo: This will be replaced by a 'selected area'
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
        if 'ZSensor' in channels:
            key = 'ZSensor'
        elif 'Topography' in channels:
            key = 'Topography'
        file_data = channels[key].data
        x_size = channels[key].xreal
        y_size = channels[key].yreal
        return file_data, (x_size, y_size)

    def _find_all_medians(self):
        # TODO: Do this only on 'selected area'. Doing it globally 
        # does not work on modulation orders higher than 1
        medians = np.ones((self.data.shape[0], self.data.shape[1]))
        medians[2,:] = 2
        for line_nr in range(0, len(self.data)):
            line = self.data[line_nr][:][:]
            median = np.median(line)
            medians[line_nr,:] = median
        return medians

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

    def _index_to_area(self, x_l, x_r, y_t, y_b):
        low_left = (
            (1e6 * x_l * self.size[0] / self.data.shape[1]),
            (self.data.shape[0] - y_b) * 1e6 * self.size[1] / self.data.shape[0],
        )
        top_right = (
            (1e6 * x_r * self.size[0] / self.data.shape[1]),
            (self.data.shape[0] - y_t) * 1e6 * self.size[1] / self.data.shape[0],
        )
        return (low_left, top_right)

    def _find_rupture_points(self, line, plot=False):
        # We know this line has exactly two jumps => Dynp is a good model
        # jump=3 seems a good compromise between robustness and not
        # missing too much data
        algo = rpt.Dynp(model='l2', min_size=20, jump=3).fit(line)
        result = algo.predict(n_bkps=2)

        start_fit = result[0]
        # result[-1] is pr definition the last point
        end_fit = result[-2]
        print('start: {}, end: {}'.format(start_fit, end_fit))

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            X = np.arange(0, len(line))
            ax.plot(X, line, 'r+', label='Data')
            ax.vlines(start_fit, line.min(), line.max())
            ax.vlines(end_fit, line.min(), line.max())
            plt.show()
        return start_fit, end_fit

    def _find_axis_endpoints(self, axis):
        found_endpoints = {}
        if axis == 'x':
            center = self.data.shape[1] // 2
        else:
            center = self.data.shape[0] // 2
        for line_nr in range(center - 20, center + 20, 3):
            if axis == 'x':
                line = self.data[line_nr, :]
            else:
                line = self.data[:, line_nr]
            start, end = self._find_rupture_points(line, plot=False)
            if (start, end) in found_endpoints:
                found_endpoints[(start, end)] += 1
            else:
                found_endpoints[(start, end)] = 1

        max_count = 0
        for end_points, count in found_endpoints.items():
            if count > max_count:
                actual_endpoints = end_points
                max_count = count
        return actual_endpoints

    def find_modulated_area(self, plot=False):
        # TODO: MULTIPROCESSING!!!!!!!!!!!!
        # These fits can be done in parallel
        found_endpoints_x = self._find_axis_endpoints('x')
        found_endpoints_y = self._find_axis_endpoints('y')
        area = self._index_to_area(
            found_endpoints_x[0],
            found_endpoints_x[1],
            found_endpoints_y[0],
            found_endpoints_y[1],
        )
        return area

    def apply_data_reset(self):
        self.data = self.original_data
        self.treatments = []
        return True

    def apply_plane_fit(self, area=None, mask=False, plot=True):
        fitted_plane = self._plane_fit(area, mask=mask, plot=plot)
        self.treatments.append('Subtract global plane fit')
        self.data = self.data - fitted_plane
        return True

    def apply_median_alignment(self):
        medians = self._find_all_medians()
        self.treatments.append('Median alignment')
        self.data = self.data - medians
        return True

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
                values.append(fit_params.get(parameter))
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
            fit_functions.sine_error_func,
            p0[:],
            args=(X.flatten(), data.flatten()),
            **FIT_PARAMS
        )
        print(fit)

        init_data = np.ones((data.shape[0], data.shape[1]))
        for j in np.arange(0, data.shape[1]):
            init_data[:, j] = fit_functions.sine_fit_func(p0, j)

        fit_data = np.ones((data.shape[0], data.shape[1]))
        for j in range(0, data.shape[1]):
            fit_data[:, j] = fit_functions.sine_fit_func(fit.x, j)

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
    # - Make line fit much more robust
    # - Remove referenes to patterned-non-modulates-area
    # - Handle rotation automatically
    # - Deal correctly with non-square images
    
    # - Significantly increase test-coverage
    # - Fix y-axis on pdf export of fits to ensure shared y-axis
    # - FFT of residuals on line fits
    # - Estimate uncertainty on roughness
    # - Establish unit-tests to keep regressions in check

    # FITTER = SPMFitter('F1.002.gwy')
    # FITTER = SPMFitter('10_40_29_WR_sin2n_500nm_20px_15x10um_20nm_1050C.gwy')
    # FITTER = SPMFitter('sample_images/camilla_thesis_afm.gwy')
    FITTER = SPMFitter('sample_images/camilla_thesis_nf.gwy')
    
    # FITTER.apply_plane_fit()
    area = FITTER.find_modulated_area(plot=True)
    # print(area)

    
    #print(FITTER._index_to_area(26.5, 814.5, 49, 1118))
    
    
    # FITTER.apply_median_alignment()
    #exit()
    #FITTER.apply_plane_fit()
    

    #print('Modulated: ', area)
    #print('Patterned: ', FITTER.find_patterned_area(plot=False))
    # FITTER.sinosodial_fit_area(area=area, plot=True)
