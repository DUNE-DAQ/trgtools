"""
Plotter with common plots to put on a single PDF.
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import numpy as np


class PDFPlotter:
    """
    Plotter with common plots to put on a single PDF.

    Using this class requires the matplotlib module.
    It creates a PdfPages object for shared plot saving
    and has a few common plots that can be used.

    The common plots take a dictionary of plot identifiers,
    such as title, xlabel, and ylabel, and a sub-dictionary
    that is used for modifying the style of the plots.

    Plotting methods require a 'title', 'xlabel', and 'ylabel'
    is within the passed plotting style dictionary. An error is
    raised if any of the three are missing.
    """

    _DEFAULT_FIG_SIZE = (6, 4)

    _DEFAULT_HIST_STYLE = dict(
            linear=True,
            linear_style=dict(color='#63ACBE', alpha=0.6, label='Linear'),
            log=True,
            log_style=dict(color='#EE442F', alpha=0.6, label='Log'),
            bins=100
    )

    _DEFAULT_ERRORBAR_STYLE = dict(
            capsize=4,
            color='k',
            ecolor='r',
            marker='.',
            markersize=0.01
    )

    def __init__(self, save_name: str) -> None:
        """
        Inits the PdfPages with the given :save_name:.

        Parameters:
            save_name (str): Initializes the PdfPages object to save to with the given name.

        Returns:
            Nothing.
        """
        if save_name.endswith('.pdf'):
            self._pdf = PdfPages(save_name)
        else:
            self._pdf = PdfPages(f"{save_name}.pdf")

        return None

    def _check_title_and_labels(self, plot_details_dict) -> None:
        """
        Check that the given :plot_details_dict: contains a title
        and labels.

        Parameter:
            plot_details_dict (dict): Dictionary containing details on plotting styles.

        Raises an error if missing.
        """
        if 'title' not in plot_details_dict:
            raise KeyError("Missing 'title' in plot_details_dict!")
        if 'xlabel' not in plot_details_dict:
            raise KeyError("Missing 'xlabel' in plot_details_dict!")
        if 'ylabel' not in plot_details_dict:
            raise KeyError("Missing 'ylabel' in plot_details_dict!")

        return None

    def get_pdf(self) -> PdfPages:
        """
        Returns the PdfPages object.

        This is useful for appending plots that are outside of this class.
        """
        return self._pdf

    def plot_histogram(
            self,
            data: np.ndarray,
            plot_details_dict: dict) -> None:
        """
        Plot a histogram onto the PdfPages object.

        Parameters:
            data (np.ndarray): Array to plot a histogram of.
            plot_details_dict (dict): Dictionary with keys such as 'title', 'xlabel', etc.

        Returns:
            Nothing. Mutates :self._pdf: with the new plot.
        """
        self._check_title_and_labels(plot_details_dict)

        plt.figure(figsize=plot_details_dict.get('figsize', self._DEFAULT_FIG_SIZE))
        ax = plt.gca()

        # Custom xticks are for specific typing. Expect to see much
        # smaller plots, so only do linear and use less bins.
        if 'xticks' in plot_details_dict:
            plt.xticks(**plot_details_dict['xticks'])

        hist_style = plot_details_dict.get('hist_style', self._DEFAULT_HIST_STYLE)
        bins = plot_details_dict.get('bins', self._DEFAULT_HIST_STYLE['bins'])

        if plot_details_dict.get('linear', True) and plot_details_dict.get('log', True):
            ax.hist(data, bins=bins, **plot_details_dict.get('linear_style', self._DEFAULT_HIST_STYLE['linear_style']))
            ax.set_yscale('linear')

            ax2 = ax.twinx()
            ax2.hist(data, bins=bins, **plot_details_dict.get('log_style', self._DEFAULT_HIST_STYLE['log_style']))
            ax2.set_yscale('log')

            # Setting the plot order
            ax.set_zorder(2)
            ax.patch.set_visible(False)
            ax2.set_zorder(1)

            linear_color = plot_details_dict.get('linear_style', self._DEFAULT_HIST_STYLE).get('color', self._DEFAULT_HIST_STYLE['linear_style']['color'])
            # Set axis and tick colors.
            ax.spines['left'].set_color(linear_color)
            ax.yaxis.label.set_color(linear_color)
            ax.tick_params('y', colors=linear_color)

            log_color = plot_details_dict.get('log_style', self._DEFAULT_HIST_STYLE).get('color', self._DEFAULT_HIST_STYLE['log_style']['color'])
            ax.spines['right'].set_color(log_color)  # Actually belongs to ax and not ax2.
            ax2.yaxis.label.set_color(log_color)
            ax2.tick_params('y', which='both', colors=log_color)

            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles + handles2
            labels = labels + labels2
            plt.legend(handles=handles, labels=labels)
        else:
            hist_style = plot_details_dict.get('linear_style', self._DEFAULT_HIST_STYLE['linear_style'])
            if plot_details_dict.get('log', self._DEFAULT_HIST_STYLE['log']):
                hist_style = plot_details_dict.get('log_style', self._DEFAULT_HIST_STYLE['log_style'])
            plt.hist(data, bins=bins, **hist_style)
            if plot_details_dict.get('log', False):  # Default to linear, so only change on log
                plt.yscale('log')

        plt.title(plot_details_dict['title'])
        ax.set_xlabel(plot_details_dict['xlabel'])
        if 'xlim' in plot_details_dict:
            plt.xlim(plot_details_dict['xlim'])

        if plot_details_dict.get('use_integer_xticks', False):
            ax.xaxis.set_major_locator(MultipleLocator(base=1))

        plt.tight_layout()
        self._pdf.savefig()
        plt.close()

        return None

    def plot_errorbar(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            plot_details_dict: dict) -> None:
        """
        Plot a scatter plot for the given x and y data to the PdfPages object.

        Parameters:
            x_data (np.ndarray): Array to use as x values.
            y_data (np.ndarray): Array to use as y values.
            plot_details_dict (dict): Dictionary with keys such as 'title', 'xlabel', etc.

        Returns:
            Nothing. Mutates :self._pdf: with the new plot.

        Error bars are handled by :plot_details_dict: since they are a style
        choice.
        """
        self._check_title_and_labels(plot_details_dict)

        plt.figure(figsize=plot_details_dict.get('figsize', self._DEFAULT_FIG_SIZE))

        errorbar_style = plot_details_dict.get('errorbar_style', self._DEFAULT_ERRORBAR_STYLE)
        plt.errorbar(x_data, y_data, **errorbar_style)

        plt.title(plot_details_dict['title'])
        plt.xlabel(plot_details_dict['xlabel'])
        plt.ylabel(plot_details_dict['ylabel'])

        plt.legend()

        plt.tight_layout()
        self._pdf.savefig()
        plt.close()
        return None

    def close(self) -> None:
        """
        Close the PdfPages object.
        """
        self._pdf.close()
        return
