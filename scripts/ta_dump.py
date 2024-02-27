#!/usr/bin/env python
"""
Display diagnostic information for TAs for a given
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


def find_save_name(run_id: int, file_index: int, overwrite: bool) -> str:
    """
    Find a new save name or overwrite an existing one.

    Parameters:
        run_id (int): The run number for the read file.
        file_index (int): The file index for the run number of the read file.
        overwrite (bool): Overwrite the 0th plot directory of the same naming.

    Returns:
        (str): Save name to write as.

    This is missing the file extension. It's the job of the save/write command
    to append the extension.
    """
    # Try to find a new name.
    name_iter = 0
    save_name = f"ta_{run_id}-{file_index:04}_figures_{name_iter:04}"

    # Outputs will always create a PDF, so use that as the comparison.
    while not overwrite and os.path.exists(save_name + ".pdf"):
        name_iter += 1
        save_name = f"ta_{run_id}-{file_index:04}_figures_{name_iter:04}"
    print(f"Saving outputs to ./{save_name}.*")

    return save_name


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
        plt.xticks(**plot_details_dict['xticks'])
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


def plot_all_event_displays(tp_data: list[np.ndarray], run_id: int, file_index: int, seconds: bool = False) -> None:
    """
    Plot all event displays.

    Parameters:
        tp_data (list[np.ndarray]): List of TPs for each TA.
        run_id (int): Run number.
        file_index (int): File index of this run.
        seconds (bool): If True, plot using seconds as units.

    Saves event displays to a single PDF.
    """
    time_unit = 's' if seconds else 'Ticks'

    with PdfPages(f"event_displays_{run_id}.{file_index:04}.pdf") as pdf:
        for tadx, ta in enumerate(tp_data):
            if seconds:
                ta = ta * TICK_TO_SEC_SCALE
            plt.figure(figsize=(6, 4))

            plt.scatter(ta['time_peak'], ta['channel'], c='k', s=2)

            # Auto limits were too wide; this narrows it.
            max_time = np.max(ta['time_peak'])
            min_time = np.min(ta['time_peak'])
            time_diff = max_time - min_time

            # Only change the xlim if there is more than one TP.
            if time_diff != 0:
                plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))

            plt.title(f'Run {run_id}.{file_index:04} Event Display: {tadx:03}')
            plt.xlabel(f"Peak Time ({time_unit})")
            plt.ylabel("Channel")

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
        ta_data: np.ndarray,
        tp_data: list[np.ndarray],
        pdf: PdfPages,
        time_label: str,
        logarithm: bool) -> None:
    """
    Plot the different time delta histograms to a PdfPages.

    Parameters:
        ta_data (np.ndarray): Array of TA data members.
        tp_data (list[np.ndarray]): List of TPs per TA. tp_data[i] holds TP data for the i-th TA.
        pdf (PdfPages): PdfPages object to append plot to.
        time_label (str): Time label to plot with (ticks vs seconds).
        logarithm (bool): Use logarithmic scaling if true.

    Returns:
        Nothing. Mutates :pdf: with the new plot.
    """
    direct_diff = ta_data['time_end'] - ta_data['time_start']
    last_tp_start_diff = []
    last_tp_peak_diff = []
    for idx, tp in enumerate(tp_data):
        last_tp_start_diff.append(np.max(tp['time_start']) - ta_data[idx]['time_start'])
        last_tp_peak_diff.append(np.max(tp['time_peak']) - ta_data[idx]['time_start'])

    last_tp_start_diff = np.array(last_tp_start_diff)
    last_tp_peak_diff = np.array(last_tp_peak_diff)

    # Seconds case.
    if "Ticks" not in time_label:
        direct_diff = direct_diff * TICK_TO_SEC_SCALE
        last_tp_start_diff = last_tp_start_diff * TICK_TO_SEC_SCALE
        last_tp_peak_diff = last_tp_peak_diff * TICK_TO_SEC_SCALE

    bins = 40

    plt.figure(figsize=(6, 4))

    plt.hist(
            (direct_diff, last_tp_start_diff, last_tp_peak_diff),
            bins=bins,
            label=(
                "TA(End) - TA(Start)",
                "Last TP(Start) - TA(Start)",
                "Last TP(Peak) - TA(Start)"
            ),
            color=(
                "#B2182B",
                "#3BB27A",
                "#2166AC"
            ),
            alpha=0.6
    )

    if logarithm:
        plt.yscale('log')

    plt.title("Time Difference Histograms")
    plt.xlabel(time_label)
    plt.legend(framealpha=0.4)

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    return None


def write_summary_stats(data: np.ndarray, filename: str, title: str) -> None:
    """
    Writes the given summary statistics to :filename:.

    Parameters:
        data (np.ndarray): Array of a TA data member.
        filename (str): File to append outputs to.
        title (str): Title of the TA data member.

    Appends statistics to the given file.
    """
    # Algorithm, Det ID, etc. are not expected to vary.
    # Check first that they don't vary, and move on if so.
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all TAs. Skipping summary statistics.")
        return None

    summary = stats.describe(data)
    std = np.sqrt(summary.variance)
    with open(filename, 'a') as out:
        out.write(f"{title}\n")
        out.write(f"Reference Statistics:\n"
                  f"\tTotal # TAs = {summary.nobs},\n"
                  f"\tMean = {summary.mean:.2f},\n"
                  f"\tStd = {std:.2f},\n"
                  f"\tMin = {summary.minmax[0]},\n"
                  f"\tMax = {summary.minmax[1]}.\n")
        std3_count = np.sum(data > summary.mean + 3*std) + np.sum(data < summary.mean - 3*std)
        std2_count = np.sum(data > summary.mean + 2*std) + np.sum(data < summary.mean - 2*std)
        out.write(f"Anomalies:\n"
                  f"\t# of >3 Sigma TAs = {std3_count},\n"
                  f"\t# of >2 Sigma TAs = {std2_count}.\n")
        out.write("\n\n")

    return None


def parse():
    """
    Parses CLI input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display diagnostic information for TAs for a given tpstream file."
    )
    parser.add_argument(
        "filename",
        help="Absolute path to tpstream file to display."
    )
    parser.add_argument(
        "--verbose", '-v',
        action="count",
        help="Increment the verbose level (errors, warnings, all)."
        "Save names and skipped writes are always printed. Default: 0.",
        default=0
    )
    parser.add_argument(
        "--no-displays",
        action="store_true",
        help="Stops the processing of event displays."
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
        help="Fragment index to stop processing (i.e. not inclusive). Takes negative indexing. Default: N.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite old outputs. Default: False."
    )

    return parser.parse_args()


def main():
    """
    Drives the processing and plotting.
    """
    # Process Arguments & Data
    args = parse()
    filename = args.filename
    verbosity = args.verbose
    no_displays = args.no_displays
    start_frag = args.start_frag
    end_frag = args.end_frag
    no_anomaly = args.no_anomaly
    seconds = args.seconds
    overwrite = args.overwrite

    linear = args.linear
    log = args.log

    # User didn't pass either flag, so default to both being true.
    if (not linear) and (not log):
        linear = True
        log = True

    data = trgtools.TAReader(filename, verbosity)

    # Load all case.
    if start_frag == 0 and end_frag == -1:
        data.read_all_fragments()  # Has extra debug/warning info
    else:  # Only load some.
        if end_frag != 0:  # Python doesn't like [n:0]
            frag_paths = data.get_fragment_paths()[start_frag:end_frag]
        elif end_frag == 0:
            frag_paths = data.get_fragment_paths()[start_frag:]

        for path in frag_paths:
            data.read_fragment(path)

    # Find a new save name or overwrite an old one.
    save_name = find_save_name(data.run_id, data.file_index, overwrite)

    print(f"Number of TAs: {data.ta_data.shape[0]}")  # Enforcing output for useful metric

    # Plotting

    if not no_anomaly:
        anomaly_filename = f"{save_name}.txt"
        print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new ta_anomaly_summary.txt
            os.remove(anomaly_filename)

    time_label = "Time (s)" if seconds else "Time (Ticks)"

    # Dictionary containing unique title, xlabel, and xticks (only some)
    plot_dict = {
            'adc_integral': {
                'title': "ADC Integral",
                'xlabel': "ADC Integral"
            },
            'adc_peak': {
                'title': "ADC Peak",
                'xlabel': "ADC Count"
            },
            'algorithm': {
                'title': "Algorithm",
                'xlabel': 'Algorithm Type',
                'bins': 8,
                'xlim': (-0.5, 7.5),
                'xticks': {
                    'ticks': range(0, 8),  # xticks to change
                    'labels': (
                        "Unknown",
                        "Supernova",
                        "Prescale",
                        "ADCSimpleWindow",
                        "HorizontalMuon",
                        "MichelElectron",
                        "DBSCAN",
                        "PlaneCoincidence"
                    ),
                    'rotation': 60,
                    'ha': 'right'  # Horizontal alignment
                }
            },
            # TODO: Channel data members should bin on
            # the available channels; however, this is
            # inconsistent between detectors (APA/CRP).
            # Requires loading channel maps.
            'channel_end': {
                'title': "Channel End",
                'xlabel': "Channel Number"
            },
            'channel_peak': {
                'title': "Channel Peak",
                'xlabel': "Channel Number"
            },
            'channel_start': {
                'title': "Channel Start",
                'xlabel': "Channel Number"
            },
            'detid': {
                'title': "Detector ID",
                'xlabel': "Detector IDs"
            },
            'num_tps': {
                'title': "Number of TPs per TA",
                'xlabel': "Number of TPs"
            },
            'time_activity': {
                'title': "Relative Time Activity",
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
                'title': "Type",
                'xlabel': "Type",
                'bins': 3,
                'xlim': (-0.5, 2.5),
                'xticks': {
                    'ticks': (0, 1, 2),  # Ticks to change
                    'labels': ('Unknown', 'TPC', 'PDS'),
                    'rotation': 60,
                    'ha': 'right'  # Horizontal alignment
                }
            },
            'version': {
                    'title': "Version",
                    'xlabel': "Versions"
            }
    }

    with PdfPages(f"{save_name}.pdf") as pdf:

        # Generic Plots
        for ta_key in data.ta_data.dtype.names:
            if 'time' in ta_key:  # Special case.
                time = data.ta_data[ta_key]
                if seconds:
                    time = time * TICK_TO_SEC_SCALE
                min_time = np.min(time)  # Prefer making the relative time change.
                plot_pdf_histogram(time - min_time, plot_dict[ta_key], pdf, linear, log)
                if not no_anomaly:
                    write_summary_stats(time - min_time, anomaly_filename,
                                        plot_dict[ta_key]['title'])
                continue

            plot_pdf_histogram(data.ta_data[ta_key], plot_dict[ta_key], pdf, linear, log)
            if not no_anomaly:
                write_summary_stats(data.ta_data[ta_key], anomaly_filename,
                                    plot_dict[ta_key]['title'])

        # Analysis Plots
        # ==== Time Delta Comparisons =====
        if linear:
            plot_pdf_time_delta_histograms(data.ta_data, data.tp_data, pdf, time_label, False)
        if log:
            plot_pdf_time_delta_histograms(data.ta_data, data.tp_data, pdf, time_label, True)
        # =================================

        # ==== Time Spans Per TC ====
        time_peak = data.ta_data['time_peak']
        time_end = data.ta_data['time_end']
        time_start = data.ta_data['time_start']
        ta_min_time = np.min((time_peak, time_end, time_start))

        time_peak -= ta_min_time
        time_end -= ta_min_time
        time_start -= ta_min_time

        if seconds:
            ta_min_time = ta_min_time * TICK_TO_SEC_SCALE
            time_peak = time_peak * TICK_TO_SEC_SCALE
            time_end = time_end * TICK_TO_SEC_SCALE
            time_start = time_start * TICK_TO_SEC_SCALE

        yerr = np.array([time_peak - time_start, time_end - time_peak]).astype(np.int64)
        time_unit = "Seconds" if seconds else "Ticks"
        time_spans_dict = {
                'title': "TA Relative Time Spans",
                'xlabel': "TA",
                'ylabel': time_label,
                'errorbar_style': {
                    'yerr': yerr,
                    'capsize': 4,
                    'color': 'k',
                    'ecolor': 'r',
                    'label': f"Avg {time_unit} / TA: {(time_peak[-1] - time_peak[0]) / len(time_peak):.2f}",
                    'marker': '.',
                    'markersize': 0.01
                }
        }
        ta_count = np.arange(len(time_peak))
        plot_pdf_errorbar(ta_count, time_peak, time_spans_dict, pdf)
        # ===========================

    if not no_displays:
        plot_all_event_displays(data.tp_data, data.run_id, data.file_index, seconds)


if __name__ == "__main__":
    main()
