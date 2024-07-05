import matplotlib.pyplot as plt

from matplotlib.widgets import Button, RectangleSelector
from matplotlib.patches import Rectangle

from spm_fitter import SPMFitter


class SPMPlotter:
    def __init__(self, filename):
        self.fitter = SPMFitter(filename)
        self.latest_select = None
        self.axes = {}
        self.rectangles = {}

        self.fig, self.axes['main'] = plt.subplots()
        self.fig.subplots_adjust(left=0.02)
        self.fig.subplots_adjust(bottom=0.05)
        self.fig.subplots_adjust(top=0.98)
        self.fig.subplots_adjust(right=0.68)
        # self.fig.subplots_adjust(wspace=0.25)
        # self.fig.subplots_adjust(hspace=0.25)
        # self._init_plot_areas()

    def _init_plot_areas(self):
        """
        Make placeholders for the two rectangles showing the patterened
        and the masked areas. Also init the rectangle selector.
        """
        kwargs = {'linewidth': 1, 'edgecolor': 'k', 'visible': False}

        self.rectangles['patterned'] = Rectangle(
            (0, 0), 1, 1, alpha=0.5, facecolor='r', **kwargs
        )
        self.axes['main'].add_patch(self.rectangles['patterned'])
        self.axes['set_patterned'] = self.fig.add_axes([0.82, 0.95, 0.1, 0.025])
        self.b_pattern = Button(self.axes['set_patterned'], 'Mark pattern')
        self.b_pattern.on_clicked(self._mark_area)

        self.rectangles['modulated'] = Rectangle(
            (0, 0), 1, 1, alpha=0.7, facecolor='c', **kwargs
        )
        self.axes['main'].add_patch(self.rectangles['modulated'])
        self.axes['set_modulated'] = self.fig.add_axes([0.82, 0.9, 0.1, 0.025])
        self.b_modulated = Button(self.axes['set_modulated'], 'Mark modulated')
        self.b_modulated.on_clicked(self._mark_area)

        self.selector = RectangleSelector(
            self.axes['main'],
            self._select_callback,
            useblit=True,
            button=[1, 3],  # disable middle button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
        )

    def _select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        area = (x2 - x1) * (y2 - y1)
        if area < 0.1:
            print('Area too small - not selected')
            self.latest_select = None
        else:
            self.latest_select = ((x1, y1), (x2, y2))
        print(self.latest_select)

    def _mark_area(self, event, rect=None, area=None):
        if event:
            if event.inaxes == self.axes['set_patterned']:
                rect = self.rectangles['patterned']
                self.fitter.patterend_region = self.latest_select
            if event.inaxes == self.axes['set_modulated']:
                rect = self.rectangles['modulated']
                self.fitter.modulated_region = self.latest_select
        if area:
            self.latest_select = area
            if rect in ('patterned', 'modulated'):
                rect = self.rectangles[rect]
                self.fitter.modulated_region = self.latest_select
                print(self.latest_select)

        width = self.latest_select[1][0] - self.latest_select[0][0]
        height = self.latest_select[1][1] - self.latest_select[0][1]
        rect.set_visible(False)
        self.fig.canvas.draw()
        rect.set_xy(self.latest_select[0])
        rect.set_width(width)
        rect.set_height(height)

    def _show_boxes(self, event):
        for rect in self.rectangles.values():
            rect.set_visible(not rect.get_visible())
        self.fig.canvas.draw()

    def _calculate_roughness(self, event):
        area = self.latest_select
        print('Calculate roughness, area is: ', area)
        print(self.fitter.calculate_roughness(area))

    def _fit_lines(self, event):
        area = self.latest_select
        print('Fit lines, area is: ', area)
        self.fitter.fit_to_all_lines(parameter='offset', area=area, plot=True)

    def _fit_area(self, event):
        plot = event.button > 1
        area = self.latest_select
        print('Fit area, area is: ', area)
        self.fitter.sinosodial_fit_area(area=area, plot=plot)

    def _reset_data(self, event):
        self.fitter.apply_data_reset()
        self.main_plot.set_data(self.fitter.data)
        min_val, max_val = self.fitter.data.min(), self.fitter.data.max()
        self.main_plot.set_clim(min_val, max_val)
        self.color_bar.mappable.set_clim(min_val, max_val)
        self.fig.canvas.draw()

    def _median_alignment(self, event):
        self.fitter.apply_median_alignment()
        self.main_plot.set_data(self.fitter.data)
        min_val, max_val = self.fitter.data.min(), self.fitter.data.max()
        self.main_plot.set_clim(min_val, max_val)
        self.color_bar.mappable.set_clim(min_val, max_val)
        self.fig.canvas.draw()

    def _plane_fit(self, event):
        # Left button: Fit selected area, right button: fit ouside selected area
        mask = event.button > 1

        area = self.latest_select
        self.fitter.apply_plane_fit(area, mask=mask)
        self.main_plot.set_data(self.fitter.data)
        min_val, max_val = self.fitter.data.min(), self.fitter.data.max()
        self.main_plot.set_clim(min_val, max_val)
        self.color_bar.mappable.set_clim(min_val, max_val)

        # ticks = np.linspace(min_val, max_val, num=6, endpoint=True)
        # self.color_bar.set_ticks(ticks)
        self.fig.canvas.draw()

    def plot_data(self):
        ax_extent = (0, self.fitter.size[0] * 1e6, 0, self.fitter.size[1] * 1e6)
        self.main_plot = self.axes['main'].imshow(
            self.fitter.data,
            interpolation='none',
            origin='upper',
            extent=ax_extent,
        )
        self.color_bar = self.fig.colorbar(self.main_plot)
        self._init_plot_areas()

        self._mark_area(
            event=None, rect='modulated', area=self.fitter.find_modulated_area()
        )
        #self._mark_area(
        #    event=None, rect='patterned', area=self.fitter.find_patterned_area()
        #)

        # Action buttons:
        self.axes['reset_data'] = self.fig.add_axes([0.82, 0.75, 0.1, 0.025])
        b_data_reset = Button(self.axes['reset_data'], 'Reset data')
        b_data_reset.on_clicked(self._reset_data)

        self.axes['show_boxes'] = self.fig.add_axes([0.82, 0.7, 0.1, 0.025])
        b_show_boxes = Button(self.axes['show_boxes'], 'Show marked')
        b_show_boxes.on_clicked(self._show_boxes)

        self.axes['calc_roughtness'] = self.fig.add_axes([0.82, 0.65, 0.1, 0.025])
        b_calc_rough = Button(self.axes['calc_roughtness'], 'Calculate roughness')
        b_calc_rough.on_clicked(self._calculate_roughness)

        self.axes['median_alignment'] = self.fig.add_axes([0.82, 0.60, 0.1, 0.025])
        b_median_alignment = Button(self.axes['median_alignment'], 'Median alignment')
        b_median_alignment.on_clicked(self._median_alignment)

        self.axes['plane_fit'] = self.fig.add_axes([0.82, 0.55, 0.1, 0.025])
        b_plane_fit = Button(self.axes['plane_fit'], 'Plane fit')
        b_plane_fit.on_clicked(self._plane_fit)

        self.axes['fit_lines'] = self.fig.add_axes([0.82, 0.50, 0.1, 0.025])
        b_fit_lines = Button(self.axes['fit_lines'], 'Fit lines')
        b_fit_lines.on_clicked(self._fit_lines)

        self.axes['area_fit'] = self.fig.add_axes([0.82, 0.45, 0.1, 0.025])
        b_fit_area = Button(self.axes['area_fit'], 'Fit area')
        b_fit_area.on_clicked(self._fit_area)

        plt.show()


if __name__ == "__main__":
    # PLOTTER = SPMPlotter("F1.002.gwy")
    PLOTTER = SPMPlotter('10_40_29_WR_sin2n_500nm_20px_15x10um_20nm_1050C.gwy')

    PLOTTER.plot_data()
