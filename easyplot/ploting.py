"""
This project is a collection of tools developed during my master thesis to
make scientific plotting easier. The main idea is to implement very few tools
which one uses often and make them easily accessible, with minimization of
ink to data ratio in mind.
"""

from datetime import datetime
import warnings
import os

# 3rd party module imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class Plot(object):
    """
    This is the main class which can be instantiated by a matplotlib.pyplot
    figure and axes objects or a figure and gridspec objects.
    >>>fig = plt.figure()
    ...ax = plt.axes()
    ...plot = Plot(fig, ax)
    one can now use the `plot.ax()` object as one would use a normal
    matplotlib.pyplot axes object the advantage is that one has a sensible
    defaults
    """
    def __init__(self):
        self.fig: plt.figure = plt.figure()
        self.ax: plt.axes = plt.axes()
        # self.fig.add_axes(ax)
        self.ax.tick_params(pad=0, length=3, labelsize=9)

    @classmethod
    def wit_existing_figure(cls, fig: plt.figure, ax: plt.axes):
        plot = cls()
        plot.fig = fig
        plot.fig.add_axes(ax)
        return plot

    @classmethod
    def gridspec(cls, fig: plt.figure, grid_spec: gridspec.GridSpec):

        ax = fig.add_subplot(grid_spec)
        return cls(fig, ax)

    def set_ticks(self, **kwargs):
        _set_ticks(self.ax, **kwargs)
        return self

    def labels(self, **kwargs):
        _labels(self.ax, **kwargs)
        return self

    def despine(self, sides='all'):
        _despine(self.ax, sides)
        return self

    def color_bar(self, image=None, despined=True, **kwargs):
        # TODO docstring

        if image is None:
            # TODO make this more general, might not work if current axes
            # holds more than one image
            im = self.ax.get_images()[0]
        else:
            im = image

        clim_min, clim_max = kwargs.get('clim', (None, None))
        min_tick, max_tick, n_ticks = kwargs.get('cticks', (None, None, None))
        cbar = plt.colorbar(im, ax=self.ax, aspect=10)
        cbar.set_clim(clim_min, clim_max)
        if (min_tick is not None and
            max_tick is not None and
            n_ticks  is not None):
            gap = (max_tick - min_tick)/(n_ticks)
            ticks = np.linspace(min_tick - gap, max_tick + gap , n_ticks + 2)

            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(np.round(tick, 2)) for tick in ticks])
        print(cbar.get_clim())
        cbar.ax.tick_params(pad=0, length=2, labelsize=7)
        cbar.update_ticks()
        cbar.outline.set_visible(not despined)
        return self

    def save(self, **kwargs):
        path = kwargs.get('path', '')
        name = kwargs.get('name', 'default_name')

        _if_not_exist_mkdir(path)
        fig_name = _make_filename(path, name)

        self.fig.savefig(fig_name, **kwargs)

def _set_ticks(axis, **kwargs):
    # TODO docstring
    if len(kwargs.items()) == 0:
        plt.setp(axis, xticks=[], xticklabels='', yticks=[], yticklabels='')
    else:
        plt.setp(axis, **kwargs)


def _labels(axis, **kwargs):
    # TODO docstring
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    plot_label = kwargs.get('plot_label', None)
    fs = kwargs.get('fontsize', 10)

    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)
    if x_label is not None:
        axis.set_xlabel(x_label, fontsize=fs)
    if y_label is not None:
        axis.set_ylabel(y_label, fontsize=fs)
    if plot_label is not None:
        axis.set_label(plot_label)

    anchor = kwargs.get('legend_anchor', None)
    ncol = kwargs.get('legend_columns', 1)

    axis.legend(bbox_to_anchor=anchor, ncol=ncol, frameon=False)


def _despine(axis, sides):
    """
    despines an axes provided a list of sides
    :param axis: instance of a pyplot Axes
    :type axis: Axes
    :param sides: either 'all' or a list of sides to despine.
        Accepted are subsets of:
        >>> ['top', 'bottom', 'left', 'right']
        statement
        >>> 'all'
        will despine every axis
    :type sides: list
    :return: returns a pyplot Axes with deleted sidelines
    :rtype Axes
    """
    if sides is 'all' or None:
        for side in ['top', 'bottom', 'left', 'right']:
            axis.spines[side].set_visible(False)
    else:
        for side in sides:
            axis.spines[side].set_visible(False)


# under development
###################

def matrix_rel_dif(m1, m2):
    r"""
    this function generates a relative difference matrix
    .. math::
       r_{i,j} = \frac{m1_{i,j}-m2_{i,j}}{m1_{i,j}}

    and plots the heatmap of this matrix. :math:`\bf m1` and :math:`\bf m1`
    must be two equally sized numpy arrays.
    :param m1: first matrix
    :type: numpy.ndarray
    :param m2: second matrix
    :type: numpy.ndarray
    :return: matplotlib.figure.Figure
    """

    _two_matrix_size_comparison_and_typecheck(m1, m2)
    s: tuple = np.shape(m1)

    abs_dif = m1 - m2
    rel_dif = abs_dif/m1

    fig = plt.figure()
    ax = plt.subplot(111)
    im = ax.imshow(rel_dif, interpolation='Nearest', origin='lower')
    ax.set_xticks(np.arange(s[0]))
    ax.set_yticks(np.arange(s[0]))
    plt.colorbar(im)
    return fig, ax


def matrix_scatter_comparison(m1, m2):
    r"""
    generates a scatter plot of elements of two matrices :math:`\bf m1` and
    :math:`\bf m1`. Which must be two equally sized numpy arrays.

    :param m1: first matrix
    :type: numpy.ndarray
    :param m2: second matrix
    :type: numpy.ndarray
    :return: matplotlib.figure.Figure
    """

    _two_matrix_size_comparison_and_typecheck(m1, m2)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(m1, m2)
    ax.plot([0, 1], color='k', linestyle='dashed')

    return fig, ax

#  some useful functions
########################


def _string_is_empty_or_spaces(string: str):
    return (string is None) or (string == '') or (string.isspace())


def _if_not_exist_mkdir(path: str):
    if _string_is_empty_or_spaces(path):
        warnings.warn('passed path is empty or contains only spaces this might '
                      'lead to errors')
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def _make_filename(path: str = None, name: str = 'empty'):

    if _string_is_empty_or_spaces(path):
        return name
    elif path.strip(' ').endswith('/'):
        return '{0}{1}'.format(path, name)
    else:
        return '{0}/{1}'.format(path, name)


def _timestamp():
    return str(datetime.now()).replace(" ", "_")


def _two_matrix_size_comparison_and_typecheck(m1, m2):
    if np.shape(m1) != np.shape(m2):
        raise ValueError('Shape mismatch. Matrices m1 and m2 must '
                         'have same shapes')
    elif type(m1) != np.ndarray:
        raise TypeError('m1 must be a numpy.ndarray not a {0}'.format(type(m1)))
    elif type(m2) != np.ndarray:
        raise TypeError('m2 must be a numpy.ndarray not a {0}'.format(type(m2)))


def main():
    pass


if __name__ == '__main__':
    pass

