"""
Display diagnostic information for TCs for a given
tpstream file.
"""

import trgtools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import os
import argparse


TICK_TO_SEC_SCALE = 16e-9  # s per tick


def plot_pdf_histogram(
        data: np.ndarray,
        plot_details_dict: dict,
        pdf: PdfPages,
        linear: bool = True,
        log: bool = True) -> None:
    """
    Plot a histogram for the given data to a PdfPages object.

    Parameters:
        data (np.ndarray): Array to plot a histogram of.
        plot_details_dict (dict): Dictionary with keys such as 'title', 'xlabel', etc.
        pdf (PdfPages): The PdfPages object that this plot will be appended to.
        linear (bool): Option to use linear y-scale.
        log (bool): Option to use logarithmic y-scale.

    Returns:
        Nothing. Mutates :pdf: with the new plot.
    """
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    bins = 100

    # Custom xticks are for specific typing. Expect to see much
    # smaller plots, so only do linear and use less bins.
    if 'xticks' in plot_details_dict:
        linear = True
        log = False
        plt.xticks(
                plot_details_dict['xticks']['ticks'],
                plot_details_dict['xticks']['labels'],
                rotation=plot_details_dict['xticks']['rotation'],
                ha=plot_details_dict['xticks']['ha']
        )
    if 'bins' in plot_details_dict:
        bins = plot_details_dict['bins']

    if linear and log:
        ax.hist(data, bins=bins, color='#63ACBE', label='Linear', alpha=0.6)
        ax.set_yscale('linear')

        ax2 = ax.twinx()
        ax2.hist(data, bins=bins, color='#EE442F', label='Log', alpha=0.6)
        ax2.set_yscale('log')

        # Setting the plot order
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax2.set_zorder(1)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles + handles2
        labels = labels + labels2
        plt.legend(handles=handles, labels=labels)
    else:
        plt.hist(data, bins=bins, color='k')
        if log:  # Default to linear, so only change on log
            plt.yscale('log')

    plt.title(plot_details_dict['title'] + " Histogram")
    ax.set_xlabel(plot_details_dict['xlabel'])
    if 'xlim' in plot_details_dict:
        plt.xlim(plot_details_dict['xlim'])

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    return None


def plot_pdf_scatter(
        x_data: np.ndarray,
        y_data: np.ndarray,
        plot_details_dict: dict,
        pdf: PdfPages) -> None:
    """
    Plot a scatter plot for the given x and y data to a PdfPages object.

    Parameters:
        x_data (np.ndarray): Array to use as x values.
        y_data (np.ndarray): Array to use as y values.
        plot_details_dict (dict): Dictionary with keys such as 'title', 'xlabel', etc.
        pdf (PdfPages): The PdfPages object that this plot will be appended to.

    Returns:
        Nothing. Mutates :pdf: with the new plot.
    """
    # May or may not have a preferred style on the scatter, e.g., marker, color, size.
    scatter_style = plot_details_dict.get('scatter_style', {})

    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, **scatter_style)

    # Asserts that the following need to be in the plotting details.
    # Who wants unlabeled plots?
    plt.title(plot_details_dict['title'])
    plt.xlabel(plot_details_dict['xlabel'])
    plt.ylabel(plot_details_dict['ylabel'])

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    return None


def plot_pdf_errorbar(
        x_data: np.ndarray,
        y_data: np.ndarray,
        plot_details_dict: dict,
        pdf: PdfPages) -> None:
    """
    Plot a scatter plot for the given x and y data to a PdfPages object.
    Error bars are handled by :plot_details_dict: since they are a style
    choice.

    Parameters:
        x_data (np.ndarray): Array to use as x values.
        y_data (np.ndarray): Array to use as y values.
        plot_details_dict (dict): Dictionary with keys such as 'title', 'xlabel', etc.
        pdf (PdfPages): The PdfPages object that this plot will be appended to.

    Returns:
        Nothing. Mutates :pdf: with the new plot.
    """
    plt.figure(figsize=plot_details_dict.get('figsize', (6, 4)))

    errorbar_style = plot_details_dict.get('errorbar_style', {})
    plt.errorbar(x_data, y_data, **errorbar_style)

    # Asserts that the following need to be in the plotting details.
    # Who wants unlabeled plots?
    plt.title(plot_details_dict['title'])
    plt.xlabel(plot_details_dict['xlabel'])
    plt.ylabel(plot_details_dict['ylabel'])

    plt.legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    return None


def plot_pdf_time_delta_histograms(
        tc_data: np.ndarray,
        ta_data: list[np.ndarray],
        pdf: PdfPages,
        time_label: str) -> None:
    """
    Plot the different time delta histograms to a PdfPages.

    Parameters:
        tc_data (np.ndarray): Array of TC data members.
        ta_data (list[np.ndarray]): List of TAs per TC. ta_data[i] holds TA data for the i-th TC.
        pdf (PdfPages): PdfPages object to append plot to.
        time_label (str): Time label to plot with (ticks vs seconds).

    Returns:
        Nothing. Mutates :pdf: with the new plot.
    """
    direct_diff = tc_data['time_end'] - tc_data['time_start']
    last_ta_start_diff = []
    last_ta_end_diff = []
    pure_ta_diff = []
    for idx, ta in enumerate(ta_data):
        last_ta_start_diff.append(np.max(ta['time_start']) - tc_data[idx]['time_start'])
        last_ta_end_diff.append(np.max(ta['time_end']) - tc_data[idx]['time_start'])
        pure_ta_diff.append(np.max(ta['time_end']) - np.min(ta['time_start']))

    last_ta_start_diff = np.array(last_ta_start_diff)
    last_ta_end_diff = np.array(last_ta_end_diff)
    pure_ta_diff = np.array(pure_ta_diff)

    # Seconds case.
    if "Ticks" not in time_label:
        direct_diff = direct_diff * TICK_TO_SEC_SCALE
        last_ta_start_diff = last_ta_start_diff * TICK_TO_SEC_SCALE
        last_ta_end_diff = last_ta_end_diff * TICK_TO_SEC_SCALE
        pure_ta_diff = pure_ta_diff * TICK_TO_SEC_SCALE

    bins = 40

    plt.figure(figsize=(6, 4))

    plt.hist(
            (direct_diff, last_ta_start_diff, last_ta_end_diff, pure_ta_diff),
            bins=bins,
            label=(
                "TC(End) - TC(Start)",
                "Last TA(Start) - TC(Start)",
                "Last TA(End) - TC(Start)",
                "Last TA(End) - First TA(Start)"
            ),
            color=(
                "#CA0020",
                "#F4A582",
                "#92C5DE",
                "#0571B0"
            ),
            alpha=0.6
    )

    plt.title("Time Difference Histograms")
    plt.xlabel(time_label)
    plt.legend(framealpha=0.4)

    plt.tight_layout()
    pdf.savefig()

    plt.close()
    return None


def write_summary_stats(data: np.ndarray, filename: str, title: str) -> None:
    """
    Writes the given summary statistics to 'filename'.

    Parameters:
        data (np.ndarray): Data set to calculate statistics on.
        filename (str): File to write to.
        title (str): Title of this data set.
    Returns:
        Nothing. Appends to the given filename.
    """
    # Algorithm, Det ID, etc. are not expected to vary.
    # Check first that they don't vary, and move on if so.
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all TCs. Skipping summary statistics.")
        return

    summary = stats.describe(data)
    std = np.sqrt(summary.variance)
    with open(filename, 'a') as out:
        out.write(f"{title}\n")
        out.write(f"Reference Statistics:\n"
                  f"\tTotal # TCs = {summary.nobs},\n"
                  f"\tMean = {summary.mean:.2f},\n"
                  f"\tStd = {std:.2f},\n"
                  f"\tMin = {summary.minmax[0]},\n"
                  f"\tMax = {summary.minmax[1]}.\n")
        std3_count = np.sum(data > summary.mean + 3*std) + np.sum(data < summary.mean - 3*std)
        std2_count = np.sum(data > summary.mean + 2*std) + np.sum(data < summary.mean - 2*std)
        out.write(f"Anomalies:\n"
                  f"\t# of >3 Sigma TCs = {std3_count},\n"
                  f"\t# of >2 Sigma TCs = {std2_count}.\n")
        out.write("\n\n")


def parse():
    """
    Parses CLI input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display diagnostic information for TCs for a given tpstream file."
    )
    parser.add_argument(
        "filename",
        help="Absolute path to tpstream file to display."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Stops the output of printed information. Default: False."
    )
    parser.add_argument(
        "--start-frag",
        type=int,
        help="Starting fragment index to process from. Takes negative indexing. Default: -10.",
        default=-10
    )
    parser.add_argument(
        "--end-frag",
        type=int,
        help="Fragment index to stop processing (i.e. not inclusive)."
             + "Takes negative indexing. Default: 0.",
        default=0
    )
    parser.add_argument(
        "--no-anomaly",
        action="store_true",
        help="Pass to not write 'ta_anomaly_summary.txt'. Default: False."
    )
    parser.add_argument(
        "--seconds",
        action="store_true",
        help="Pass to use seconds instead of time ticks. Default: False."
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Pass to use linear histogram scaling. Default: plots both linear and log."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Pass to use logarithmic histogram scaling. Default: plots both linear and log."
    )

    return parser.parse_args()


def main():
    """
    Drives the processing and plotting.
    """
    # Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet
    start_frag = args.start_frag
    end_frag = args.end_frag
    no_anomaly = args.no_anomaly
    seconds = args.seconds
    linear = args.linear
    log = args.log

    # User didn't pass either flag, so default to both being true.
    if (not linear) and (not log):
        linear = True
        log = True

    data = trgtools.TCData(filename, quiet)
    run_id = data.run_id
    file_index = data.file_index

    # Load all case.
    if start_frag == 0 and end_frag == -1:
        data.load_all_frags()  # Has extra debug/warning info
    else:  # Only load some.
        if end_frag != 0:  # Python doesn't like [n:0]
            frag_paths = data.get_tc_frag_paths()[start_frag:end_frag]
        elif end_frag == 0:
            frag_paths = data.get_tc_frag_paths()[start_frag:]

        # Does not count empty frags.
        for path in frag_paths:
            data.load_frag(path)

    print(f"Number of TCs: {data.tc_data.shape[0]}")  # Enforcing output for useful metric

    # Plotting

    anomaly_filename = f"tc_anomalies_{run_id}-{file_index:04}.txt"
    if not no_anomaly:
        if not quiet:
            print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new ta_anomaly_summary.txt
            os.remove(anomaly_filename)

    time_label = "Time (s)" if seconds else "Time (Ticks)"

    # Dictionary containing unique title, xlabel, and xticks (only some)
    plot_dict = {
            'algorithm': {
                'bins': np.arange(-0.5, 9.5, 1),
                'title': "Algorithm",
                'xlabel': 'Algorithm Type',
                'xlim': (-1, 9),
                'xticks': {
                    'ticks': range(0, 9),
                    'labels': (
                        "Unknown",
                        "Supernova",
                        "HSIEventToTriggerCandidate",
                        "Prescale",
                        "ADCSimpleWindow",
                        "HorizontalMuon",
                        "MichelElectron",
                        "PlaneCoincidence",
                        "Custom"
                    ),
                    'rotation': 60,
                    'ha': 'right'  # Horizontal alignment
                }
            },
            'detid': {
                'title': "Detector ID",
                'xlabel': "Detector IDs"
            },
            'num_tas': {
                'title': "Number of TAs per TC",
                'xlabel': "Number of TAs"
            },
            'time_candidate': {
                'title': "Relative Time Candidate",
                'xlabel': time_label
            },
            'time_end': {
                'title': "Relative Time End",
                'xlabel': time_label
            },
            'time_peak': {
                'title': "Relative Time Peak",
                'xlabel': time_label
            },
            'time_start': {
                'title': "Relative Time Start",
                'xlabel': time_label
            },
            'type': {
                'bins': np.arange(-0.5, 10.5, 1),
                'title': "Type",
                'xlabel': "Type",
                'xlim': (-1, 10),
                'xticks': {
                    'ticks': range(0, 10),  # Ticks to change
                    'labels': (
                        'Unknown',
                        'Timing',
                        'TPCLowE',
                        'Supernova',
                        'Random',
                        'Prescale',
                        'ADCSimpleWindow',
                        'HorizontalMuon',
                        'MichelElectron',
                        'PlaneCoincidence'
                        ),
                    'rotation': 60,
                    'ha': 'right'  # Horizontal alignment
                    }
            },
            'version': {
                'title': "Version",
                'xlabel': "Versions"
            }
    }
    with PdfPages(f"tc_data_member_histograms_{run_id}-{file_index:04}.pdf") as pdf:

        # Generic plots
        for tc_key in data.tc_data.dtype.names:
            if 'time' in tc_key:  # Special case.
                time = data.tc_data[tc_key]
                if seconds:
                    time = time * TICK_TO_SEC_SCALE
                min_time = np.min(time)  # Prefer making the relative time change.
                plot_pdf_histogram(time - min_time, plot_dict[tc_key], pdf, linear, log)
                if not no_anomaly:
                    write_summary_stats(time - min_time, anomaly_filename,
                                        plot_dict[tc_key]['title'])
                continue

            plot_pdf_histogram(data.tc_data[tc_key], plot_dict[tc_key], pdf, linear, log)
            if not no_anomaly:
                write_summary_stats(data.tc_data[tc_key], anomaly_filename,
                                    plot_dict[tc_key]['title'])

        # "Analysis" plots
        # ==== Time Delta Comparisons =====
        plot_pdf_time_delta_histograms(data.tc_data, data.ta_data, pdf, time_label)

        tc_adc_integrals = np.array([np.sum(tas['adc_integral']) for tas in data.ta_data])
        adc_integrals_dict = {
                'title': "TC ADC Integrals",
                'xlabel': "ADC Integral"
        }
        plot_pdf_histogram(tc_adc_integrals, adc_integrals_dict, pdf, linear, log)
        # =================================

        # ==== ADC Integral vs Number of TAs ====
        integral_vs_num_tas_dict = {
                'title': "TC ADC Integral vs Number of TAs",
                'xlabel': "Number of TAs",
                'ylabel': "TC ADC Integral",
                'scatter_style': {
                    'alpha': 0.6,
                    'c': 'k',
                    's': 2
                }
        }
        plot_pdf_scatter(data.tc_data['num_tas'], tc_adc_integrals, integral_vs_num_tas_dict, pdf)
        # =======================================

        # ==== Time Spans Per TC ====
        time_candidate = data.tc_data['time_candidate']
        time_end = data.tc_data['time_end']
        time_start = data.tc_data['time_start']
        tc_min_time = np.min((time_candidate, time_end, time_start))

        time_candidate -= tc_min_time
        time_end -= tc_min_time
        time_start -= tc_min_time

        if seconds:
            tc_min_time = tc_min_time * TICK_TO_SEC_SCALE
            time_candidate = time_candidate * TICK_TO_SEC_SCALE
            time_end = time_end * TICK_TO_SEC_SCALE
            time_start = time_start * TICK_TO_SEC_SCALE

        yerr = np.array([time_candidate - time_start, time_end - time_candidate]).astype(np.int64)
        time_unit = "Seconds" if seconds else "Ticks"
        time_spans_dict = {
                'title': "TC Relative Time Spans",
                'xlabel': "TC",
                'ylabel': time_label,
                'errorbar_style': {
                    'yerr': yerr,
                    'capsize': 4,
                    'color': 'k',
                    'ecolor': 'r',
                    'label': f"Avg {time_unit} / TC: {(time_candidate[-1] - time_candidate[0]) / len(time_candidate):.2f}",
                    'marker': '.',
                    'markersize': 0.01
                }
        }
        tc_count = np.arange(len(time_candidate))
        plot_pdf_errorbar(tc_count, time_candidate, time_spans_dict, pdf)
        # ===========================

    return None


if __name__ == "__main__":
    main()
