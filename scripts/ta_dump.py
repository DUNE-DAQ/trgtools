#!/usr/bin/env python
"""
Display diagnostic information for TAs for a given
tpstream file.
"""
import trgtools
from trgtools.plot import PDFPlotter

import trgdataformats

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import os
import argparse


ALGORITHM_LABELS = list(trgdataformats.TriggerActivityData.Algorithm.__members__.keys())
ALGORITHM_TICKS = [ta_alg.value for ta_alg in trgdataformats.TriggerActivityData.Algorithm.__members__.values()]
TYPE_LABELS = list(trgdataformats.TriggerActivityData.Type.__members__.keys())
TYPE_TICKS = [ta_type.value for ta_type in trgdataformats.TriggerActivityData.Type.__members__.values()]

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


def plot_all_event_displays(tp_data: list[np.ndarray], ta_data: np.ndarray, run_id: int, file_index: int, seconds: bool = False) -> None:
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
        for tadx, (ta_event, tp_event) in enumerate(zip(ta_data, tp_data)):
            if seconds:
                tp_event['time_start'] = tp_event['time_start'] * TICK_TO_SEC_SCALE
            plt.figure(figsize=(6, 4))

            times = tp_event['time_start'] - np.min(tp_event['time_start'])
            plt.scatter(times, tp_event['channel'], c='k', s=2)

            # Auto limits were too wide; this narrows it.
            max_time = np.max(times)
            min_time = np.min(times)
            time_diff = max_time - min_time

            # Only change the xlim if there is more than one TP.
            if time_diff != 0:
                plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))

            plt.title(f"Run {run_id}.{file_index:04} Event Display: {tadx:03} \n trigger record: {ta_event['trigger_number']}")
            plt.xlabel(f"Relative Start Time ({time_unit})")
            plt.ylabel("Channel")

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
        max_peak_arg = np.argmax(tp['samples_to_peak'])

        # FIXME: Replace the hard-coded SOT to TOT scaling.
        max_peak_time = tp[max_peak_arg]['samples_to_peak'].astype(np.uint64) * 32 + tp[max_peak_arg]['time_start']

        last_tp_peak_diff.append(max_peak_time +  - ta_data[idx]['time_start'])

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
        description="Display diagnostic information for TAs for a given HDF5 file."
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
    parser.add_argument(
        "--batch-mode", "-b",
        action="store_true",
        help="Do you want to run in the batch mode (without loading bars/tqdm)?"
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
    batch_mode = args.batch_mode

    linear = args.linear
    log = args.log

    # User didn't pass either flag, so default to both being true.
    if (not linear) and (not log):
        linear = True
        log = True

    data = trgtools.TAReader(filename, verbosity, batch_mode)

    # Check that there are TA fragments.
    if len(data.get_fragment_paths()) == 0:
        print("File doesn't contain any TriggerActivity fragments.")
        return 1

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
    plot_hist_dict = {
            'adc_integral': {
                'title': "ADC Integral Histogram",
                'xlabel': "ADC Integral",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'adc_peak': {
                'title': "ADC Peak Histogram",
                'xlabel': "ADC Count",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'algorithm': {
                'bins': np.sort(np.array([(tick-0.45, tick+0.45) for tick in ALGORITHM_TICKS]).flatten()),
                'title': "Algorithm Histogram",
                'xlabel': 'Algorithm Type',
                'ylabel': "Count",
                'linear': True,  # TODO: Hard set for now.
                'linear_style': dict(color='k'),
                'log': False,
                'xticks': {
                        'labels': ALGORITHM_LABELS,
                        'ticks': ALGORITHM_TICKS,
                        'fontsize': 6,
                        'rotation': 60,
                        'ha': 'right'  # Horizontal alignment
                    }
            },
            # TODO: Channel data members should bin on
            # the available channels; however, this is
            # inconsistent between detectors (APA/CRP).
            # Requires loading channel maps.
            'channel_end': {
                'title': "Channel End Histogram",
                'xlabel': "Channel Number",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'channel_peak': {
                'title': "Channel Peak Histogram",
                'xlabel': "Channel Number",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'channel_start': {
                'title': "Channel Start Histogram",
                'xlabel': "Channel Number",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'detid': {
                'title': "Detector ID Histogram",
                'xlabel': "Detector IDs",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            },
            'num_tps': {
                'title': "Number of TPs per TA Histogram",
                'xlabel': "Number of TPs",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_activity': {
                'title': "Relative Time Activity Histogram",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_end': {
                'title': "Relative Time End Histogram",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_peak': {
                'title': "Relative Time Peak Histogram",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_start': {
                'title': "Relative Time Start Histogram",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'type': {
                'bins': np.sort(np.array([(tick-0.45, tick+0.45) for tick in TYPE_TICKS]).flatten()),
                'title': "Type Histogram",
                'xlabel': "Type",
                'ylabel': "Count",
                'linear': True,  # TODO: Hard set for now.
                'linear_style': dict(color='k'),
                'log': False,
                'xticks': {
                        'labels': TYPE_LABELS,
                        'ticks': TYPE_TICKS,
                        'fontsize': 6,
                        'rotation': 60,
                        'ha': 'right'  # Horizontal alignment
                    }
            },
            'trigger_number': {
                'title': "Trigger Number Histogram",
                'xlabel': "Trigger Number",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'version': {
                'title': "Version Histogram",
                'xlabel': "Versions",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            }
    }

    pdf_plotter = PDFPlotter(f"{save_name}.pdf")
    # Generic Plots
    for ta_key in data.ta_data.dtype.names:
        if 'time' in ta_key:  # Special case.
            time = data.ta_data[ta_key]
            if seconds:
                time = time * TICK_TO_SEC_SCALE
            min_time = np.min(time)  # Prefer making the relative time change.
            pdf_plotter.plot_histogram(time - min_time, plot_hist_dict[ta_key])
            if not no_anomaly:
                write_summary_stats(time - min_time, anomaly_filename, ta_key)
            continue

        if ta_key == 'algorithm' or ta_key == 'type':  # Special case.
            plot_data = np.array([datum.value for datum in data.ta_data[ta_key]], dtype=int)
            pdf_plotter.plot_histogram(plot_data, plot_hist_dict[ta_key])
            if not no_anomaly:
                write_summary_stats(plot_data, anomaly_filename, ta_key)
            del plot_data
            continue

        pdf_plotter.plot_histogram(data.ta_data[ta_key], plot_hist_dict[ta_key])
        if not no_anomaly:
            write_summary_stats(data.ta_data[ta_key], anomaly_filename, ta_key)

    # Analysis Plots
    pdf = pdf_plotter.get_pdf()  # Needed for extra plots that are not general.
    # ==== Time Delta Comparisons =====
    if linear:
        plot_pdf_time_delta_histograms(data.ta_data, data.tp_data, pdf, time_label, False)
    if log:
        plot_pdf_time_delta_histograms(data.ta_data, data.tp_data, pdf, time_label, True)
    # =================================

    # ==== Time Spans Per TA ====
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
                'ecolor': "#EE442F",
                'label': f"Avg {time_unit} / TA: {(time_peak[-1] - time_peak[0]) / len(time_peak):.2f}",
                'mec': "#EE442F",
                'mfc': "#EE442F",
                'marker': 'h',
                'markersize': 4.00
            }
    }
    ta_count = np.arange(len(time_peak))
    pdf_plotter.plot_errorbar(ta_count, time_peak, time_spans_dict)
    # ===========================
    pdf_plotter.close()

    if not no_displays:
        plot_all_event_displays(data.tp_data, data.ta_data, data.run_id, data.file_index, seconds)

    return None


if __name__ == "__main__":
    main()
