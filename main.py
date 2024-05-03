import gwyfile
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, RectangleSelector


class SPMFitter:
    def __init__(self, filename):
        # Raw data as read from file
        self.original_data, self.size = self._load_file(filename)
        self.patterend_region = None
        self.modulated_region = None
        self.latest_select = None

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

    def find_all_medians(self):
        row_sum = np.zeros(len(self.data[1]))
        for i in range(0, len(self.data[1])):
            row_sum[i] = sum(self.data[:][i])
        fig, ax = plt.subplots()
        ax.plot(row_sum)
        plt.show()

    def mask_patterned_area(self, tl, br):
        x1 = tl[0]
        y1 = tl[1]
        x2 = br[0]
        y2 = br[1]
        masked = self.data
        print(y1, y2, x1, x2)
        masked[y1:y2, x1:x2] = 0
        fig, ax = plt.subplots()
        ax.imshow(
            masked,
            interpolation='none',
            origin='upper',
            extent=(0, self.size[0], 0, self.size[1]),
        )
        plt.show()

    def calculate_roughness(self, area=None):
        # todo: RMS is just one way of calculating surface roughness
        # consider to implement other techniques

        # If not none, data is the relevant subset rather than all data
        if area is None:
            data = self.data

        square_sum = np.square(data).sum()
        square_mean = square_sum / data.size
        rms = np.sqrt(square_mean)
        return rms

    def _plane_fit(self, plot=False):
        data = self.data
        # https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
        X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        G = np.ones((data.shape[0] * data.shape[1], 3))
        G[:, 0] = X.flatten()
        G[:, 1] = Y.flatten()
        Z = data.flatten()

        (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
        # Begin magic, too long since I went to school....
        normal = (a, b, -1)
        nn = np.linalg.norm(normal)
        normal = normal / nn
        point = np.array([0.0, 0.0, c])
        d = -point.dot(normal)
        z = (-normal[0] * X - normal[1] * Y - d) * 1.0 / normal[2]
        # End of magic

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(X, Y, data, rstride=10, cstride=10)
            ax.plot_surface(X, Y, z, rstride=10, cstride=10, alpha=0.2)
            ax.plot_surface(X, Y, data - z, rstride=10, cstride=10, alpha=0.2)
            plt.show()
        return z

    def apply_plane_fit(self):
        fitted_plane = self._plane_fit()
        self.treatments.append('Subtract global plane fit')
        self.data = self.data - fitted_plane
        return True

    def fit_line(self, line_number: int, plot=False):
        # TODO: THIS SHOULD BE DERIVED FROM DATA - NOT HARD CODED!!
        line = self.data[line_number][:][50:450]
        # line = self.data[line_number][:][:]
        X = np.arange(0, len(line))

        peaks, _ = sp.signal.find_peaks(line, distance=20)

        # A decent guess for frequency is to distribute all peaks acress the line
        freq_guess = len(peaks) * 2 * np.pi / len(line)
        # We fit a sine, zero should be a quater period offset
        phase_guess = peaks[0] - (peaks[1] - peaks[0]) / 4

        # Amplitude guess is simply max minus min
        z_mean = sum(line) / len(line)
        ampl_guess = max(line) - z_mean
        p0 = [ampl_guess, freq_guess, phase_guess, z_mean, 0]

        fitfunc = lambda p, x: p[0] * np.sin(p[1] * x + p[2]) + p[3] + p[4] * x
        errfunc = lambda p, x, y: fitfunc(p, x) - y

        params = {
            'method': 'lm',
            'jac': '2-point',
            'ftol': 1e-13,
            'xtol': 1e-13,
            'max_nfev': 20000,
        }
        fit = sp.optimize.least_squares(errfunc, p0[:], args=(X, line), **params)

        if plot:
            print(fit)
            fig, ax = plt.subplots()
            # for peak in peaks:
            #    ax.plot(peak, line[peak], 'bo')
            ax.plot(X, line, 'r+', label='Data')
            # ax.plot(X, fitfunc(p0, X), label='Initial function')
            ax.plot(X, fitfunc(fit.x, X), label='Fitted function')
            plt.legend()
            plt.show()

        fit_params = None
        if fit.status > 0:
            fit_params = {
                'amplitude': fit.x[0],
                'frequency': fit.x[1],
                'phase': fit.x[2],
                'offset': fit.x[3],
                'slope': fit.x[4],
                'iterations': fit.nfev,
            }
        return fit_params

    def fit_to_all_lines(self, parameter, plot=False):
        values = []
        for line in range(0, self.data.shape[0]):
            fit = self.fit_line(line)
            if fit is None:
                values.append(None)
            else:
                values.append(fit[parameter])

        if plot:
            fig, ax = plt.subplots()
            ax.plot(values, 'r+', label=parameter)
            plt.show()
        return values

    def plot_data(self):
        def select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            area = (x2 - x1) * (y2 - y1)
            print(area)
            if area < 1:
                self.latest_select = None
            else:
                self.latest_select = ((x1, y1), (x2, y2))

        def mark_pattern(event):
            print('Mark pattern: {}'.format(self.latest_select))
            self.patterend_region = self.latest_select

        def mark_modulated(event):
            print('Mark modulated: {}'.format(self.latest_select))
            self.modulated_region = self.latest_select

        fig, ax = plt.subplots()

        ax.imshow(
            self.data,
            interpolation='none',
            origin='upper',
            extent=(0, self.size[0] * 1e6, 0, self.size[1] * 1e6),
        )
        selector = RectangleSelector(
            ax,
            select_callback,
            useblit=True,
            button=[1, 3],  # disable middle button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            # spancoords='data',
            interactive=True,
        )

        ax_set_pattern = fig.add_axes([0.82, 0.8, 0.1, 0.025])
        ax_set_modulated = fig.add_axes([0.82, 0.75, 0.1, 0.025])
        b_pattern = Button(ax_set_pattern, 'Mark pattern')
        b_pattern.on_clicked(mark_pattern)
        b_modulated = Button(ax_set_modulated, 'Mark modulated')
        b_modulated.on_clicked(mark_modulated)

        plt.show()


if __name__ == "__main__":
    # TODO:
    # - Find good way to read user input regarding positions to fit
    # - Plot residuals of fits

    FITTER = SPMFitter("F1.002.gwy")

    # FITTER.fit_to_all_lines('frequency', plot=True)

    # FITTER.apply_plane_fit()
    FITTER.plot_data()
    # print(FITTER.calculate_roughness())

    # fit = FITTER.fit_line(7, plot=True)
    # fit = FITTER.fit_line(262, plot=True)

    # Z = FITTER._plane_fit()
    # FITTER.data = FITTER.data - Z
    # FITTER.plot_data()
    # FITTER.mask_patterned_area( (24, 120), (476, 476) )
    # FITTER.find_all_medians()
