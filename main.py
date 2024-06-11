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
            (0, 0), 1, 1, alpha=0.5, facecolor='r', **kwargs)
        self.axes['main'].add_patch(self.rectangles['patterned'])
        self.axes['set_patterned'] = self.fig.add_axes([0.82, 0.8, 0.1, 0.025])
        self.b_pattern = Button(self.axes['set_patterned'], 'Mark pattern')
        self.b_pattern.on_clicked(self._mark_area)

        self.rectangles['modulated'] = Rectangle(
            (0, 0), 1, 1, alpha=0.7, facecolor='c', **kwargs)
        self.axes['main'].add_patch(self.rectangles['modulated'])
        self.axes['set_modulated'] = self.fig.add_axes([0.82, 0.75, 0.1, 0.025])
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

    def _mark_area(self, event):
        print('klaf')
        if event.inaxes == self.axes['set_patterned']:
            rect = self.rectangles['patterned']
            self.fitter.patterend_region = self.latest_select
        if event.inaxes == self.axes['set_modulated']:
            rect = self.rectangles['modulated']
            self.fitter.modulated_region = self.latest_select
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

    def plot_data(self):
        ax_extent = (0, self.fitter.size[0] * 1e6, 0, self.fitter.size[1] * 1e6)
        main_plot = self.axes['main'].imshow(
            self.fitter.data, interpolation='none', origin='upper', extent=ax_extent,
        )
        self.fig.colorbar(main_plot)
        self._init_plot_areas()

        self.axes['show_boxes'] = self.fig.add_axes([0.82, 0.7, 0.1, 0.025])
        b_show_boxes = Button(self.axes['show_boxes'], 'Show marked')
        b_show_boxes.on_clicked(self._show_boxes)

        self.axes['calc_roughtness'] = self.fig.add_axes([0.82, 0.65, 0.1, 0.025])
        b_calc_rough = Button(self.axes['calc_roughtness'], 'Calculate roughness')
        b_calc_rough.on_clicked(self._calculate_roughness)

        plt.show()


if __name__ == "__main__":
    PLOTTER = SPMPlotter("F1.002.gwy")

    PLOTTER.plot_data()
