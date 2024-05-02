import gwyfile
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class SPMFitter():
    def __init__(self, filename):
        self.data, self.size = self._load_file(filename)

    def _load_file(self, filename):
        # TODO: Take other formats and convert with gwyddion command line
        obj = gwyfile.load(filename)
        channels = gwyfile.util.get_datafields(obj)
        # TODO: What about the other channels
        # TODO: Is main channel always 'ZSensor'?
        file_data = channels['ZSensor'].data
        x_size = channels['ZSensor'].xreal
        y_size = channels['ZSensor'].yreal
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
        ax.imshow(masked, interpolation='none', origin='upper',
                  extent=(0, self.size[0], 0, self.size[1]))
        plt.show()

    def plane_fit(self, plot=False):
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
        z = (-normal[0]*X - normal[1]*Y - d)*1. / normal[2]
        # End of magic

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(X, Y, data, rstride=10, cstride=10)
            ax.plot_surface(X, Y, z, rstride=10, cstride=10, alpha=0.2)
            ax.plot_surface(X, Y, data - z, rstride=10, cstride=10, alpha=0.2)
            plt.show()
        return z

    def fit_line(self, line_number: int):
        line = self.data[line_number][:][50:450]
        X = np.arange(0, len(line))

        z_mean = sum(line) / len(line)
        ampl_guess = max(line) - z_mean

        fitfunc = lambda p, x: p[0] * np.sin(p[1]*x+p[2]) + p[3] + p[4] * x
        errfunc = lambda p, x, y: fitfunc(p, x) - y

        # Need to work on automated guesses for the remaining two parameters
        p0 = [ampl_guess, 0.235, -13.6, z_mean, 0]
        a = sp.optimize.least_squares(errfunc, p0[:], args=(X, line), method='lm', jac='3-point', max_nfev=20000)
        print(a)

        fig, ax = plt.subplots()
        ax.plot(X, line, 'r+', label='Data')
        # ax.plot(X, fitfunc(p0, X), label='Initial function')
        ax.plot(X, fitfunc(a.x, X), label='Fitted function')
        plt.legend()
        plt.show()

    def plot_data(self):
        fig, ax = plt.subplots()
        ax.imshow(self.data, interpolation='none', origin='upper',
                  extent=(0, self.size[0], 0, self.size[1]))

        # Plot magic values from Nolans script
        x_coord = 24 * self.size[0] / len(self.data[0])
        y_coord = self.size[1] - (120 * self.size[1] / len(self.data[1]))
        plt.plot(x_coord, y_coord, 'r+')
        # x_coord = 476 * self.size[0] / len(self.data[0])
        # y_coord = 476 * self.size[1] / len(self.data[1])
        # plt.plot(x_coord, y_coord, 'r+')
        plt.show()


if __name__ == '__main__':
    FITTER = SPMFitter('F1.002.gwy')

    FITTER.fit_line(300)

    # Z = FITTER.plane_fit()
    # FITTER.data = FITTER.data - Z
    # FITTER.plot_data()
    # FITTER.mask_patterned_area( (24, 120), (476, 476) )
    # FITTER.find_all_medians()
