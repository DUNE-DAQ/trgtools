"""
Display diagnostic information for TPs in a given
tpstream file.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import trgtools

TICK_TO_SEC_SCALE = 512e-9 # secs per tick

def channel_tot(tp_data):
    """
    Plot the TP channel vs time over threshold scatter plot.
    """
    channels = []
    tots = []
    for frag_data in tp_data:
        channel_data = frag_data['channel']
        channels = channels + list(channel_data)
        tot_data = frag_data['time_over_threshold']# * TICK_TO_SEC_SCALE
        tots = tots + list(tot_data)

    channels = np.array(channels)
    tots = np.array(tots)
    hot_channels = channels[np.where(tots >= 0.0075)]

    plt.figure(figsize=(6,4), dpi=200)

    plt.scatter(channels, tots, c='k', s=2, label='TP')

    plt.title("TP Time Over Threshold vs Channel")
    plt.xlabel("Channel")
    plt.ylabel("Time Over Threshold (Ticks)")
    plt.legend()

    #plt.annotate(f"High ToT Channels: {np.unique(hot_channels)}", xy=(750, 7250), va="center", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("channel_vs_tot.png")
    plt.close()

def tp_percent_histogram(tp_data):

    channels = []
    for frag_data in tp_data:
        channel_data = frag_data['channel']
        channels = channels + list(channel_data)
    channels = np.array(channels)
    counts, bins = np.histogram(channels, bins=np.arange(0.5, 3072.5, 1))

    count_mask = np.ones(counts.shape, dtype=bool)
    total_counts = np.sum(counts)

    percent01 = np.sum(counts > 0.01*total_counts)
    count_mask[np.where(counts > 0.01*total_counts)] = False

    percent001 = np.sum(counts[count_mask] > 0.001*total_counts)
    count_mask[np.where(counts[count_mask] > 0.001*total_counts)] = False

    percent0001 = np.sum(counts[count_mask] > 0.0001*total_counts)
    count_mask[np.where(counts[count_mask] > 0.0001*total_counts)] = False

    remaining = np.sum(counts[count_mask])

    plt.figure(figsize=(6,4))
    plt.stairs([percent01, percent001, percent0001, remaining], [0,1,2,3,4], color='k', fill=True)

    plt.xlim((0,4))
    plt.xticks([0.5, 1.5, 2.5, 3.5], ['>1%', '>0.1%', '>0.01%', "Remainder"])

    plt.title(f"Number of Channels Contributing % To Total Count {total_counts}")

    plt.tight_layout()
    plt.savefig("percent_total.svg")
    plt.close()

def tp_channel_histogram(tp_data, quiet=False):
    """
    Plot the TP channel histogram.
    """
    # tp_data is a list of length num frags
    # Each element is an array with size num TPs
    # => Iterate over every fragment -> get TP channel data -> append to list

    channels = []
    for frag_data in tp_data:
        channel_data = frag_data['channel']
        channels = channels + list(channel_data)

    counts, bins = np.histogram(channels, bins=np.arange(0.5, 3072.5, 1))
    total_counts = np.sum(counts)
    if (not quiet):
        print("High TP Count Channels:", np.where(counts >= 500))
        print("Percentage Counts:", np.where(counts >= 0.01*total_counts))

    plt.figure(figsize=(6,4))

    plt.stairs(counts, bins, fill=True, color='k', label=f'TP Count: {total_counts}')

    plt.title("TP Channel Histogram")
    plt.xlabel("Channel")
    plt.legend()

    plt.ylim((0, 800))

    plt.tight_layout()
    plt.savefig("tp_channel_histogram.svg")
    plt.close()

def plot_summary_stats(tp_data, no_anomaly=False, quiet=False):
    """
    Plot summary statistics on the various TP member data.
    Displays as box plots on multiple pages of a PDF.
    """
    # 'Sanity' titles _should_ all be the same value.
    titles = {
                'adc_integral': "ADC Integral Summary",
                'adc_peak': "ADC Peak Summary",
                'algorithm': "Algorithm (Sanity) Summary",
                'channel': "Channel Summary",
                'detid': "Detector ID (Sanity) Summary",
                'flag': "Flag (Sanity) Summary",
                'time_over_threshold': "Time Over Threshold Summary",
                'time_peak': "Time Peak Summary",
                'time_start': "Time Start Summary",
                'type': "Type (Sanity) Summary",
                'version': "Version (Sanity) Summary"
             }
    labels = {
                'adc_integral': "ADC Integral",
                'adc_peak': "ADC Count",
                'algorithm': "",
                'channel': "Channel Number",
                'detid': "",
                'flag': "",
                'time_over_threshold': "Ticks",
                'time_peak': "Ticks",
                'time_start': "Ticks",
                'type': "",
                'version': ""
             }

    anomaly_filename = 'tp_anomaly_summary.txt'

    if not no_anomaly:
        if not quiet:
            print(f"Writing descriptive statistics to {anomaly_filename}.")
        import os
        from scipy import stats # Only used when writing to file.
        if os.path.isfile(anomaly_filename):
            # Prepare a new anomaly_summary.txt
            os.remove(anomaly_filename)

    with PdfPages("tp_summary_stats.pdf") as pdf:
        for tp_key, title in titles.items():
            plot_data = []
            for frag_data in tp_data:
                plot_data = plot_data + list(frag_data[tp_key])
            plot_data = np.array(plot_data)
            plt.figure(figsize=(6,4))
            ax = plt.gca()

            plt.boxplot(plot_data, notch=True, vert=False, sym='+')
            plt.yticks([])
            ax.xaxis.grid(True)
            plt.xlabel(labels[tp_key])

            plt.title(title)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Write anomalies to file.
            if not no_anomaly:
                if "Sanity" in title and np.all(plot_data == plot_data[0]):
                    # Either passed check or all wrong in the same way.
                    continue
                summary = stats.describe(plot_data)
                std = np.sqrt(summary.variance)
                with open(anomaly_filename, 'a') as out:
                    out.write(f"{title}\n")
                    out.write(f"Reference Statistics:\n"            \
                              f"\tTotal # TPs = {summary.nobs},\n"  \
                              f"\tMean = {summary.mean:.2f},\n"     \
                              f"\tStd = {std:.2f},\n"               \
                              f"\tMin = {summary.minmax[0]},\n"     \
                              f"\tMax = {summary.minmax[1]}.\n")
                    std3_count = np.sum(plot_data > summary.mean + 3*std) \
                               + np.sum(plot_data < summary.mean - 3*std)
                    std2_count = np.sum(plot_data > summary.mean + 2*std) \
                               + np.sum(plot_data < summary.mean - 2*std)
                    out.write(f"Anomalies:\n"                           \
                              f"\t# of >3 Sigma TPs = {std3_count},\n"  \
                              f"\t# of >2 Sigma TPs = {std2_count}.\n")
                    out.write("\n\n")

def parse():
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")
    parser.add_argument("--quiet", action="store_true", help="Stops the output of printed information. Default: False.")
    parser.add_argument("--start-frag", type=int, help="Fragment to start loading from (inclusive); can take negative integers. Default: -10", default=-10)
    parser.add_argument("--end-frag", type=int, help="Fragment to stop loading at (exclusive); can take negative integers. Default: 0", default=0)
    parser.add_argument("--no-anomaly", action="store_true", help="Pass to not write 'anomaly_summary.txt. Default: False.")

    return parser.parse_args()

def main():
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet
    start_frag = args.start_frag
    end_frag = args.end_frag
    no_anomaly = args.no_anomaly

    data = trgtools.TPData(filename, quiet)
    if end_frag == 0: # Ex: [-10:0] is bad.
        frag_paths = data.get_tp_frag_paths()[start_frag:]
    else:
        frag_paths = data.get_tp_frag_paths()[start_frag:end_frag]

    for path in frag_paths:
        if (not quiet):
            print("Fragment Path:", path)
        data.load_frag(path)

    if (not quiet):
        print("Length of tp_data:", len(data.tp_data))
    tp_channel_histogram(data.tp_data, quiet)
    channel_tot(data.tp_data)
    tp_percent_histogram(data.tp_data)
    plot_summary_stats(data.tp_data, no_anomaly, quiet)

if __name__ == "__main__":
    main()
