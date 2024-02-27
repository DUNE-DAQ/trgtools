#!/usr/bin/env python
"""
Display diagnostic information for TPs in a given
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
    save_name = f"tp_{run_id}-{file_index:04}_figures_{name_iter:04}"

    # Outputs will always create a PDF, so use that as the comparison.
    while not overwrite and os.path.exists(save_name + ".pdf"):
        name_iter += 1
        save_name = f"tp_{run_id}-{file_index:04}_figures_{name_iter:04}"
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


def plot_pdf_tot_vs_channel(tp_data: np.ndarray, pdf: PdfPages) -> None:
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

    plt.scatter(tp_data['channel'], tp_data['time_over_threshold'], c='k', s=2, label='TP', rasterized=True)

    plt.title("TP Time Over Threshold vs Channel")
    plt.xlabel("Channel")
    plt.ylabel("Time Over Threshold (Ticks)")
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

    plt.figure(figsize=(6, 4), dpi=200)

    plt.scatter(
        tp_data['adc_peak'],
        tp_data['adc_integral'],
        c='k',
        s=2,
        label='TP',
        rasterized=True
    )
    plt.scatter(
        tp_data['adc_peak'][high_integral_locs],
        tp_data['adc_integral'][high_integral_locs],
        c='#63ACBE',
        s=2, marker='+',
        label=r'$2^{15}-1$',
        rasterized=True
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
        description="Display diagnostic information for TAs for a given tpstream file."
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

    linear = args.linear
    log = args.log

    # User didn't pass either flag, so default to both being true.
    if (not linear) and (not log):
        linear = True
        log = True

    data = trgtools.TPReader(filename, verbosity)

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
                'xlim': (-1, 2),
                'xticks': {
                    'ticks': (0, 1),
                    'labels': ("Unknown", "TPCDefault")
                },
                'bins': (-0.5, 0.5, 1.5)  # TODO: Dangerous. Hides values outside of this range.
            },
            # TODO: Channel should bin on the available
            # channels; however, this is inconsistent
            # between detectors (APA/CRP).
            # Requires loading channel maps.
            'channel': {
                'title': "Channel",
                'xlabel': "Channel Number"
            },
            'detid': {
                'title': "Detector ID",
                'xlabel': "Detector IDs"
            },
            'flag': {
                'title': "Flag",
                'xlabel': "Flags"
            },
            'time_over_threshold': {
                'title': "Time Over Threshold",
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
                'xlim': (-1, 3),
                'xticks': {
                    'ticks': (0, 1, 2),
                    'labels': ('Unknown', 'TPC', 'PDS')
                },
                'bins': (-0.5, 0.5, 1.5, 2.5)  # TODO: Dangerous. Hides values outside of this range.
            },
            'version': {
                'title': "Version",
                'xlabel': "Versions"
            }
    }
    with PdfPages(f"{save_name}.pdf") as pdf:

        # Generic plots
        for tp_key in data.tp_data.dtype.names:
            if 'time' in tp_key:  # Special case.
                time = data.tp_data[tp_key]
                if seconds:
                    time = time * TICK_TO_SEC_SCALE
                min_time = np.min(time)  # Prefer making the relative time change.
                plot_pdf_histogram(time - min_time, plot_dict[tp_key], pdf, linear, log)
                if not no_anomaly:
                    write_summary_stats(time - min_time, anomaly_filename,
                                        plot_dict[tp_key]['title'])
                continue

            plot_pdf_histogram(data.tp_data[tp_key], plot_dict[tp_key], pdf, linear, log)
            if not no_anomaly:
                write_summary_stats(data.tp_data[tp_key], anomaly_filename,
                                    plot_dict[tp_key]['title'])

        # Analysis plots
        # ==== Time Over Threshold vs Channel ====
        plot_pdf_tot_vs_channel(data.tp_data, pdf)
        # ========================================

        # ==== ADC Integral vs ADC Peak ====
        plot_pdf_adc_integral_vs_peak(data.tp_data, pdf, verbosity)
        # ===================================

    return None


if __name__ == "__main__":
    main()
