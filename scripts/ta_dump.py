#!/usr/bin/env python
"""
Display diagnostic information for TAs for a given
tpstream file.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import trgtools

TICK_TO_SEC_SCALE = 16e-9 # s per tick

def time_start_plot(start_times, seconds=False):
    """
    Plot TA start times vs TA index.

    Optionally, use ticks or seconds for scaling.
    """
    time_unit = 'Ticks'
    if seconds:
        start_times = start_times * TICK_TO_SEC_SCALE
        time_unit = 's'

    first_time = start_times[0]
    total_time = start_times[-1] - start_times[0]
    total_tas = start_times.shape[0]
    avg_rate = total_tas / total_time

    plt.figure(figsize=(6,4))
    plt.plot(start_times - first_time, 'k', label=f"Average Rate: {avg_rate:.3} TA/{time_unit}")

    plt.title(f"TA Start Times: Shifted by {first_time:.4e} {time_unit}")
    plt.xlabel("TA Order")
    plt.ylabel(f"Start Time ({time_unit})")

    plt.legend()

    plt.tight_layout()
    plt.savefig("ta_start_times.svg")
    plt.close()


def plot_pdf_time_delta_histograms(
        ta_data: np.ndarray,
        tp_data: list[np.ndarray],
        pdf: PdfPages,
        time_label: str) -> None:
    """
    Plot the different time delta histograms to a PdfPages.

    Parameters:
        ta_data (np.ndarray): Array of TA data members.
        tp_data (list[np.ndarray]): List of TPs per TA. tp_data[i] holds TP data for the i-th TA.
        pdf (PdfPages): PdfPages object to append plot to.
        time_label (str): Time label to plot with (ticks vs seconds).

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

    plt.title("Time Difference Histograms")
    plt.xlabel(time_label)
    plt.legend(framealpha=0.4)

    plt.tight_layout()
    pdf.savefig()

    plt.close()
    return None


def write_summary_stats(data, filename, title):
    """
    Writes the given summary statistics to 'filename'.
    """
    # Algorithm, Det ID, etc. are not expected to vary.
    # Check first that they don't vary, and move on if so.
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all TAs. Skipping summary statistics.")
        return

    summary = stats.describe(data)
    std = np.sqrt(summary.variance)
    with open(filename, 'a') as out:
        out.write(f"{title}\n")
        out.write(f"Reference Statistics:\n"            \
                  f"\tTotal # TPs = {summary.nobs},\n"  \
                  f"\tMean = {summary.mean:.2f},\n"     \
                  f"\tStd = {std:.2f},\n"               \
                  f"\tMin = {summary.minmax[0]},\n"     \
                  f"\tMax = {summary.minmax[1]}.\n")
        std3_count = np.sum(data > summary.mean + 3*std) \
                   + np.sum(data < summary.mean - 3*std)
        std2_count = np.sum(data > summary.mean + 2*std) \
                   + np.sum(data < summary.mean - 2*std)
        out.write(f"Anomalies:\n"                           \
                  f"\t# of >3 Sigma TPs = {std3_count},\n"  \
                  f"\t# of >2 Sigma TPs = {std2_count}.\n")
        out.write("\n\n")

def plot_pdf_histogram(data, plot_details_dict, pdf, linear=True, log=True):
    """
    Plot a histogram for the given data to a PdfPage object.
    """
    plt.figure(figsize=(6,4))
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

def all_event_displays(tp_data, run_id, file_index, seconds=False):
    """
    Plot all event_displays as pages in a PDF.

    Optionally, use ticks or seconds for scaling.
    """
    time_unit = 's' if seconds else 'Ticks'

    with PdfPages(f"event_displays_{run_id}.{file_index:04}.pdf") as pdf:
        for tadx, ta in enumerate(tp_data):
            if seconds:
                ta = ta * TICK_TO_SEC_SCALE
            plt.figure(figsize=(6,4))

            plt.scatter(ta['time_peak'], ta['channel'], c='k', s=2)

            # Auto limits were too wide; this narrows it.
            max_time = np.max(ta['time_peak'])
            min_time = np.min(ta['time_peak'])
            time_diff = max_time - min_time
            plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))

            plt.title(f'Run {run_id}.{file_index:04} Event Display: {tadx:03}')
            plt.xlabel(f"Peak Time ({time_unit})")
            plt.ylabel("Channel")

            plt.tight_layout()
            pdf.savefig()
            plt.close()


def event_display(peak_times, channels, idx, seconds=False):
    """
    Plot an individual event display.

    Optionally, use ticks or seconds for scaling.
    """
    time_unit = 'Ticks'
    if seconds:
        peak_times = peak_times * TICK_TO_SEC_SCALE
        time_unit = 's'

    plt.figure(figsize=(6,4))

    plt.scatter(peak_times, channels, c='k', s=2)
    max_time = np.max(peak_times)
    min_time = np.min(peak_times)
    time_diff = max_time - min_time

    plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))

    plt.title(f"Event Display: {idx:03}")
    plt.xlabel(f"Peak Time ({time_unit})")
    plt.ylabel("Channel")

    plt.tight_layout()
    plt.savefig(f"./event_display_{idx:03}.svg")
    plt.close()

def parse():
    """
    Parses CLI input arguments.
    """
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")
    parser.add_argument("--quiet", action="store_true", help="Stops the output of printed information. Default: False.")
    parser.add_argument("--no-displays", action="store_true", help="Stops the processing of event displays.")
    parser.add_argument("--start-frag", type=int, help="Starting fragment index to process from. Takes negative indexing. Default: -10.", default=-10)
    parser.add_argument("--end-frag", type=int, help="Fragment index to stop processing (i.e. not inclusive). Takes negative indexing. Default: 0.", default=0)
    parser.add_argument("--no-anomaly", action="store_true", help="Pass to not write 'ta_anomaly_summary.txt'. Default: False.")
    parser.add_argument("--seconds", action="store_true", help="Pass to use seconds instead of time ticks. Default: False.")
    parser.add_argument("--linear", action="store_true", help="Pass to use linear histogram scaling. Default: plots both linear and log.")
    parser.add_argument("--log", action="store_true", help="Pass to use logarithmic histogram scaling. Default: plots both linear and log.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite old outputs. Default: False.")

    return parser.parse_args()

def main():
    """
    Drives the processing and plotting.
    """
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet
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

    data = trgtools.TAData(filename, quiet)

    # Load all case.
    if start_frag == 0 and end_frag == -1:
        data.load_all_frags() # Has extra debug/warning info
    else: # Only load some.
        if end_frag != 0: # Python doesn't like [n:0]
            frag_paths = data.get_ta_frag_paths()[start_frag:end_frag]
        elif end_frag == 0:
            frag_paths = data.get_ta_frag_paths()[start_frag:]

        # Does not count empty frags.
        for path in frag_paths:
            data.load_frag(path)

    # Try to find an empty plotting directory
    plot_iter = 0
    plot_dir = f"{data.run_id}-{data.file_index}_figures_{plot_iter:04}"
    while not overwrite and os.path.isdir(plot_dir):
        plot_iter += 1
        plot_dir = f"{data.run_id}-{data.file_index}_figures_{plot_iter:04}"
    print(f"Saving outputs to ./{plot_dir}/")
    # If overwriting and it does exist, don't need to make it.
    # So take the inverse to mkdir.
    if not (overwrite and os.path.isdir(plot_dir)):
        os.mkdir(plot_dir)
    os.chdir(plot_dir)

    print(f"Number of TAs: {data.ta_data.shape[0]}") # Enforcing output for useful metric

    ## Plotting
    # General Data Member Plots
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
    if not no_anomaly:
        anomaly_filename = "ta_anomalies.txt"
        if not quiet:
            print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new ta_anomaly_summary.txt
            os.remove(anomaly_filename)

    with PdfPages("ta_data_member_histograms.pdf") as pdf:
        for ta_key in data.ta_data.dtype.names:
            if not no_anomaly:
                write_summary_stats(data.ta_data[ta_key], anomaly_filename, plot_dict[ta_key]['title'])
            if 'time' in ta_key:
                time = data.ta_data[ta_key]
                if seconds:
                    time = time * TICK_TO_SEC_SCALE
                min_time = np.min(time)  # Prefer making the relative time change.
                plot_pdf_histogram(time - min_time, plot_dict[ta_key], pdf, linear, log)
                continue
            plot_pdf_histogram(data.ta_data[ta_key], plot_dict[ta_key], pdf, linear, log)

        # Analysis Plots
        plot_pdf_time_delta_histograms(data.ta_data, data.tp_data, pdf, time_label)

    if (not no_displays):
        all_event_displays(data.tp_data, data.run_id, data.file_index, seconds)

if __name__ == "__main__":
    main()
