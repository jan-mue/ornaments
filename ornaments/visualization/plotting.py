import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import PolyCollection
from geometer import Point, Rectangle

from .mpl_interaction import figure_pz
from ..wallpaper_groups import group_name, calculate_visible_lattice_lines


def show_images(images, lattices=None, labels=None, points=None, polygons=None):
    n = len(images)
    k = int(np.ceil(np.sqrt(n)))
    l = int(np.ceil(n/k))

    for i, image in enumerate(images, 1):
        plt.subplot(l, k, i)
        if labels is not None:
            label = labels[i-1]
            try:
                label = group_name(label)
            except ValueError:
                pass
            plt.title("Group: " + str(label))
        if points is not None:
            pts = np.array(points[i-1])
            plt.scatter([p[0] for p in pts], [p[1] for p in pts], color='red')
        if polygons is not None:
            verts = polygons[i-1]
            poly = PolyCollection(verts, facecolors='none', edgecolors='darkgray')
            ax = plt.gca()
            ax.add_collection(poly)
        if image.shape[-1] == 1:
            plt.imshow(image[:, :, 0], cmap="gray")
        else:
            plt.imshow(image)
        if lattices is not None:
            draw_lattice(*lattices[i-1])
    plt.show()


def draw_lines(lines, ax):
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()
    viewport = Rectangle(Point(xmin, ymin), Point(xmax, ymin), Point(xmax, ymax), Point(xmin, ymax))

    for line in lines:
        a, b = viewport.intersect(line)
        a, b = a.normalized_array, b.normalized_array
        ax.add_line(mlines.Line2D([a[0], b[0]], [a[1], b[1]], color='gray'))


def draw_lattice(t0, t1, t2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()

    t0, t1, t2 = np.array(t0), np.array(t1), np.array(t2)
    lines = calculate_visible_lattice_lines(t0, t1, t2, (xmin, xmax, ymin, ymax))
    draw_lines(lines, ax)


class OrnamentPlot:

    def __init__(self, tile, width=1024, height=1024):
        self._tile = tile
        self._figure = figure_pz()
        self._axes = self._figure.add_subplot(1, 1, 1)

        extent = (0, width, 0, height)
        img = self._get_pattern(*extent)
        self._max_extent = extent
        self._image = self._axes.imshow(img)

        self._axes.callbacks.connect('xlim_changed', self._on_axis_change)
        self._axes.callbacks.connect('ylim_changed', self._on_axis_change)
        plt.show()

    def _get_pattern(self, xmin, xmax, ymin, ymax):
        h, w, c = self._tile.shape
        xtiles = 0
        xcrop = int(xmin)
        if xmax >= 0:
            xtiles += int(np.ceil(xmax / w))
        if xmin < 0:
            xtiles += int(np.ceil(-xmin / w))
            xcrop = int(np.ceil(-xmin / w)*w + xmin)

        ytiles = 0
        ycrop = int(ymin)
        if ymax >= 0:
            ytiles += int(np.ceil(ymax / h))
        if ymin < 0:
            ytiles += int(np.ceil(-ymin / h))
            ycrop = int(np.ceil(-ymin / h)*h + ymin)

        pattern = np.tile(self._tile, (ytiles, xtiles, 1))
        return pattern[ycrop:ycrop+int(ymax-ymin), xcrop:xcrop+int(xmax-xmin)]

    def _on_axis_change(self, ax):
        ax.set_autoscale_on(False)  # Otherwise, infinite loop

        # Get the range for the new area
        xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
        xend = xstart + xdelta
        yend = ystart + ydelta

        img = self._get_pattern(xstart, xend, yend, ystart)
        self._image.set_data(img)
        self._image.set_extent((xstart, xend, ystart, yend))
        self._figure.canvas.draw_idle()
