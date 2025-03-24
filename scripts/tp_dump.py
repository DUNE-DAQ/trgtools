#!/usr/bin/env python
"""
Display diagnostic information for TPs in a given
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
    save_name = f"tp_{run_id}-{file_index:04}_figures_{name_iter:04}"

    # Outputs will always create a PDF, so use that as the comparison.
    while not overwrite and os.path.exists(save_name + ".pdf"):
        name_iter += 1
        save_name = f"tp_{run_id}-{file_index:04}_figures_{name_iter:04}"
    print(f"Saving outputs to ./{save_name}.*")

    return save_name


def plot_pdf_sot_vs_channel(tp_data: np.ndarray, pdf: PdfPages) -> None:
    """
    Plot the TP channel vs time over threshold scatter plot.

    Parameter:
        tp_data (np.ndarray): Array of TPs.
        pdf (PdfPages): The PdfPages object that this plot will be appended to.

    Returns:
        Nothing. Mutates :pdf: with the new plot.

    Does not have a seconds option. This plot is more informative in the ticks version.
    """
    plt.figure(figsize=(6, 4), dpi=200)

    plt.plot(tp_data['channel'], tp_data['samples_over_threshold'], 'hk', mew=0, alpha=0.4, ms=2, label='TP', rasterized=True)

    plt.title("TP Samples Over Threshold vs Channel")
    plt.xlabel("Channel")
    plt.ylabel("Samples Over Threshold (Readout Ticks)")
    plt.legend()

    plt.tight_layout()
    pdf.savefig()  # Many scatter points makes this a PNG ._.
    plt.close()
    return None


def plot_pdf_adc_integral_vs_peak(tp_data: np.ndarray, pdf: PdfPages, verbosity: int = 0) -> None:
    """
    Plot the ADC Integral vs ADC Peak.

    Parameters:
        tp_data (np.ndarray): Array of TPs.
        pdf (PdfPages): The PdfPages object that this plot will be appended to.
        verbosity (int): Verbose level to print information.

    Returns:
        Nothing. Mutates :pdf: with the new plot.
    """
    if verbosity >= 2:
        print(
            "Number of ADC Integrals at Signed 16 Limit:",
            np.sum(tp_data['adc_integral'] == np.power(2, 15)-1)
        )
        print("Total number of TPs:", len(tp_data['adc_peak']))
    high_integral_locs = np.where(tp_data['adc_integral'] == np.power(2, 15)-1)
    unity = (np.min(tp_data['adc_peak']), np.max(tp_data['adc_peak']))

    plt.figure(figsize=(6, 4), dpi=200)

    plt.plot(
        tp_data['adc_peak'],
        tp_data['adc_integral'],
        c="#00000055",
        ms=2,
        ls='none',
        marker='o',
        mew=0,
        label='TP',
        zorder=2,
        rasterized=True
    )
    plt.scatter(
        tp_data['adc_peak'][high_integral_locs],
        tp_data['adc_integral'][high_integral_locs],
        c='#63ACBE',
        s=2, marker='+',
        label=r'$2^{15}-1$',
        zorder=3,
        rasterized=True
    )
    plt.plot(
        unity,
        unity,
        color='#EE442F',
        label="Unity",
        lw=2,
        zorder=1
    )

    plt.title("ADC Integral vs ADC Peak")
    plt.xlabel("ADC Peak")
    plt.ylabel("ADC Integral")
    plt.legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    return None


def write_summary_stats(data: np.ndarray, filename: str, title: str) -> None:
    """
    Writes the given summary statistics to :filename:.

    Parameters:
        data (np.ndarray): Array of a TP data member.
        filename (str): File to append outputs to.
        title (str): Title of the TP data member.

    Appends statistics to the given file.
    """
    # Algorithm, Det ID, etc. are not expected to vary.
    # Check first that they don't vary, and move on if so.
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all TPs. Skipping summary statistics.")
        return None

    summary = stats.describe(data)
    std = np.sqrt(summary.variance)
    with open(filename, 'a') as out:
        out.write(f"{title}\n")
        out.write(f"Reference Statistics:\n"
                  f"\tTotal # TPs = {summary.nobs},\n"
                  f"\tMean = {summary.mean:.2f},\n"
                  f"\tStd = {std:.2f},\n"
                  f"\tMin = {summary.minmax[0]},\n"
                  f"\tMax = {summary.minmax[1]}.\n")
        std3_count = np.sum(data > summary.mean + 3*std) + np.sum(data < summary.mean - 3*std)
        std2_count = np.sum(data > summary.mean + 2*std) + np.sum(data < summary.mean - 2*std)
        out.write(f"Anomalies:\n"
                  f"\t# of >3 Sigma TPs = {std3_count},\n"
                  f"\t# of >2 Sigma TPs = {std2_count}.\n")
        out.write("\n\n")

    return None


def parse():
    parser = argparse.ArgumentParser(
        description="Display diagnostic information for TPs for a given HDF5 file."
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
        help="Fragment to start loading from (inclusive); can take negative integers. Default: -10",
        default=-10
    )
    parser.add_argument(
        "--end-frag",
        type=int,
        help="Fragment to stop loading at (exclusive); can take negative integers. Default: N",
        default=0
    )
    parser.add_argument(
        "--no-anomaly",
        action="store_true",
        help="Pass to not write 'tp_anomaly_summary.txt'. Default: False."
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

    data = trgtools.TPReader(filename, verbosity, batch_mode)

    # Check that there are TP fragments.
    if len(data.get_fragment_paths()) == 0:
        print("File doesn't contain any TriggerPrimitive fragments.")
        return 1

    # Load all case
    if start_frag == 0 and end_frag == -1:
        data.read_all_fragments()  # Has extra debug/warning info
    else:
        if end_frag == 0:  # Ex: [-10:0] is bad.
            frag_paths = data.get_fragment_paths()[start_frag:]
        else:
            frag_paths = data.get_fragment_paths()[start_frag:end_frag]

        for path in frag_paths:
            data.read_fragment(path)

    # Find a new save name or overwrite an old one.
    save_name = find_save_name(data.run_id, data.file_index, overwrite)

    print(f"Number of TPs: {data.tp_data.shape[0]}")  # Enforcing output for a useful metric.

    # Plotting

    if not no_anomaly:
        anomaly_filename = f"{save_name}.txt"
        if verbosity >= 2:
            print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new tp_anomaly_summary.txt
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
            # TODO: Channel should bin on the available
            # channels; however, this is inconsistent
            # between detectors (APA/CRP).
            # Requires loading channel maps.
            'channel': {
                'title': "Channel Histogram",
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
            'flag': {
                'title': "Flag Histogram",
                'xlabel': "Flags",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log'),
                'use_integer_xticks': True
            },
            'samples_over_threshold': {
                'title': "Samples Over Threshold Histogram",
                'xlabel': time_label,
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'samples_to_peak': {
                'title': "Samples To Peak Histogram",
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

    pdf_plotter = PDFPlotter(save_name)

    # Generic plots
    for tp_key in data.tp_data.dtype.names:
        if 'time' in tp_key:  # Special case.
            time = data.tp_data[tp_key]
            if seconds:
                time = time * TICK_TO_SEC_SCALE
            min_time = np.min(time)  # Prefer making the relative time change.
            pdf_plotter.plot_histogram(time - min_time, plot_hist_dict[tp_key])
            if not no_anomaly:
                write_summary_stats(time - min_time, anomaly_filename, tp_key)
            continue

        pdf_plotter.plot_histogram(data.tp_data[tp_key], plot_hist_dict[tp_key])
        if not no_anomaly:
            write_summary_stats(data.tp_data[tp_key], anomaly_filename, tp_key)

    pdf = pdf_plotter.get_pdf()
    # Analysis plots
    # ==== Time Over Threshold vs Channel ====
    plot_pdf_sot_vs_channel(data.tp_data, pdf)
    # ========================================

    # ==== ADC Integral vs ADC Peak ====
    plot_pdf_adc_integral_vs_peak(data.tp_data, pdf, verbosity)
    # ===================================
    pdf_plotter.close()

    return None


if __name__ == "__main__":
    main()
