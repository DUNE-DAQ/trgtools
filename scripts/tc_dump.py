#!/usr/bin/env python
"""
Display diagnostic information for TCs for a given
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


ALGORITHM_LABELS = list(trgdataformats.TriggerCandidateData.Algorithm.__members__.keys())
ALGORITHM_TICKS = [tp_alg.value for tp_alg in trgdataformats.TriggerCandidateData.Algorithm.__members__.values()]
TYPE_LABELS = list(trgdataformats.TriggerCandidateData.Type.__members__.keys())
TYPE_TICKS = [tp_type.value for tp_type in trgdataformats.TriggerCandidateData.Type.__members__.values()]

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
    save_name = f"tc_{run_id}-{file_index:04}_figures_{name_iter:04}"

    # Outputs will always create a PDF, so use that as the comparison.
    while not overwrite and os.path.exists(save_name + ".pdf"):
        name_iter += 1
        save_name = f"tc_{run_id}-{file_index:04}_figures_{name_iter:04}"
    print(f"Saving outputs to ./{save_name}.*")

    return save_name


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


def plot_pdf_time_delta_histograms(
        tc_data: np.ndarray,
        ta_data: list[np.ndarray],
        pdf: PdfPages,
        time_label: str,
        logarithm: bool) -> None:
    """
    Plot the different time delta histograms to a PdfPages.

    Parameters:
        tc_data (np.ndarray): Array of TC data members.
        ta_data (list[np.ndarray]): List of TAs per TC. ta_data[i] holds TA data for the i-th TC.
        pdf (PdfPages): PdfPages object to append plot to.
        time_label (str): Time label to plot with (ticks vs seconds).
        logarithm (bool): Use logarithmic scaling if true.

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
    Writes the given summary statistics to 'filename'.

    Parameters:
        data (np.ndarray): Array of a TC data member.
        filename (str): File to append outputs to.
        title (str): Title of the TC data member.

    Appends statistics to the given file.
    """
    # Algorithm, Det ID, etc. are not expected to vary.
    # Check first that they don't vary, and move on if so.
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all TCs. Skipping summary statistics.")
        return None

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

    return None


def parse():
    """
    Parses CLI input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display diagnostic information for TCs for a given HDF5 file."
    )
    parser.add_argument(
        "filename",
        help="Absolute path to tpstream file to display."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        help="Increment the verbose level (errors, warnings, all)."
        "Save names and skipped writes are always printed. Default: 0.",
        default=0
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
        "--batch_mode", "-b",
        action="store_true",
        help="Do you want to run in batch mode (without loading bars/tqdm)?"
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

    data = trgtools.TCReader(filename, verbosity, batch_mode)

    # Check that there are TC fragments.
    if len(data.get_fragment_paths()) == 0:
        print("File doesn't contain any TriggerCandidate fragments.")
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

    print(f"Number of TCs: {data.tc_data.shape[0]}")  # Enforcing output for useful metric

    # Plotting

    if not no_anomaly:
        anomaly_filename = f"{save_name}.txt"
        if verbosity >= 2:
            print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new ta_anomaly_summary.txt
            os.remove(anomaly_filename)

    time_label = "Time (s)" if seconds else "Time (Ticks)"

    # Dictionary containing unique title, xlabel, and xticks (only some)
    plot_hist_dict = {
            'algorithm': {
                'bins': np.sort(np.array([(tick-0.45, tick+0.45) for tick in ALGORITHM_TICKS]).flatten()),
                'title': "Algorithm",
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
            'detid': {
                'title': "Detector ID",
                'xlabel': "Detector IDs",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            },
            'num_tas': {
                'title': "Number of TAs per TC",
                'xlabel': "Number of TAs",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            },
            'time_candidate': {
                'title': "Relative Time Candidate",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_end': {
                'title': "Relative Time End",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_peak': {
                'title': "Relative Time Peak",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'time_start': {
                'title': "Relative Time Start",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'type': {
                'bins': np.sort(np.array([(tick-0.45, tick+0.45) for tick in TYPE_TICKS]).flatten()),
                'title': "Type",
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
                'title': "Trigger Number",
                'xlabel': "Trigger Number",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            },
            'version': {
                'title': "Version",
                'xlabel': "Versions",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            }
    }

    pdf_plotter = PDFPlotter(save_name)

    # Generic plots
    for tc_key in data.tc_data.dtype.names:
        if 'time' in tc_key:  # Special case.
            time = data.tc_data[tc_key]
            if seconds:
                time = time * TICK_TO_SEC_SCALE
            min_time = np.min(time)  # Prefer making the relative time change.
            pdf_plotter.plot_histogram(time - min_time, plot_hist_dict[tc_key])
            if not no_anomaly:
                write_summary_stats(time - min_time, anomaly_filename, tc_key)
            continue

        if tc_key == 'algorithm' or tc_key == 'type':  # Special case.
            plot_data = np.array([datum.value for datum in data.tc_data[tc_key]], dtype=int)
            pdf_plotter.plot_histogram(plot_data, plot_hist_dict[tc_key])
            if not no_anomaly:
                write_summary_stats(plot_data, anomaly_filename, tc_key)
            del plot_data
            continue

        pdf_plotter.plot_histogram(data.tc_data[tc_key], plot_hist_dict[tc_key])
        if not no_anomaly:
            write_summary_stats(data.tc_data[tc_key], anomaly_filename, tc_key)

    pdf = pdf_plotter.get_pdf()
    # Analysis plots
    # ==== Time Delta Comparisons =====
    if np.sum(data.tc_data['num_tas']) > 0:
        if linear:
            plot_pdf_time_delta_histograms(data.tc_data, data.ta_data, pdf, time_label, False)
        if log:
            plot_pdf_time_delta_histograms(data.tc_data, data.ta_data, pdf, time_label, True)
    # =================================

    # ==== TC ADC Integrals ====
    if np.sum(data.tc_data['num_tas']) > 0:
        tc_adc_integrals = np.array([np.sum(tas['adc_integral']) for tas in data.ta_data])
        adc_integrals_dict = {
                'title': "TC ADC Integrals",
                'xlabel': "ADC Integral",
                'ylabel': "Count"
        }
        pdf_plotter.plot_histogram(tc_adc_integrals, adc_integrals_dict)
    # ==========================

    # ==== ADC Integral vs Number of TAs ====
    if np.sum(data.tc_data['num_tas']) > 0:
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
                'ecolor': "#EE442F",
                'label': f"Avg {time_unit} / TC: "
                         f"{(time_candidate[-1] - time_candidate[0]) / len(time_candidate):.2f}",
                'mec': "#EE442F",
                'mfc': "#EE442F",
                'marker': 'h',
                'markersize': 4.00
            }
    }
    tc_count = np.arange(len(time_candidate))
    pdf_plotter.plot_errorbar(tc_count, time_candidate, time_spans_dict)
    # ===========================
    pdf_plotter.close()

    return None


if __name__ == "__main__":
    main()
