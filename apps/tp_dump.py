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
from scipy import stats

import trgtools

TICK_TO_SEC_SCALE = 16e-9 # secs per tick

def channel_tot(tp_data):
    """
    Plot the TP channel vs time over threshold scatter plot.
    """
    plt.figure(figsize=(6,4), dpi=200)

    plt.scatter(tp_data['channel'], tp_data['time_over_threshold'], c='k', s=2, label='TP')

    plt.title("TP Time Over Threshold vs Channel")
    plt.xlabel("Channel")
    plt.ylabel("Time Over Threshold (Ticks)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("tp_channel_vs_tot.png") # Many scatter points makes this a PNG
    plt.close()

def tp_percent_histogram(tp_data):
    """
    Plot the count of TPs that account for 1%, 0.1%, and 0.01%
    of the total channel count.
    """
    counts, _ = np.histogram(tp_data['channel'], bins=np.arange(0.5, 3072.5, 1))

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

def plot_channel_histogram(channels, quiet=False):
    """
    Plot the TP channel histogram.
    """
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

def plot_version_histogram(versions, quiet=False):
    """
    Plot the TP versions histogram.

    Generally, this will be one value.
    """
    plt.figure(figsize=(6,4))

    plt.hist(versions, color='k')

    plt.title("TP Versions Histogram")
    plt.xlabel("Versions")

    plt.tight_layout()
    plt.savefig("tp_versions_histogram.svg")
    plt.close()

def plot_type_histogram(types, quiet=False):
    """
    Plot the TP types histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(types, bins=np.arange(-0.5, 3, 1), range=(0,2), align='mid', color='k')

    plt.title("TP Types Histogram")
    plt.xlabel("Types")
    plt.xticks([0,1,2], ["Unknown", "TPC", "PDS"]) # Taken from TriggerPrimitive.hpp

    plt.tight_layout()
    plt.savefig("tp_types_histogram.svg")
    plt.close()

def plot_time_start_histogram(times, quiet=False):
    """
    Plot the TP time starts histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(times, color='k')

    plt.title("TP Time Starts Histogram")
    plt.xlabel("Time Start (Ticks)")

    plt.tight_layout()
    plt.savefig("tp_time_start_histogram.svg")
    plt.close()

def plot_time_peak_histogram(times, quiet=False):
    """
    Plot the TP time peaks histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(times, color='k')

    plt.title("TP Time Peaks Histogram")
    plt.xlabel("Time Peak (Ticks)")

    plt.tight_layout()
    plt.savefig("tp_time_peak_histogram.svg")
    plt.close()

def plot_time_over_threshold_histogram(times, quiet=False):
    """
    Plot the TP time over threshold histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(times, color='k')

    plt.title("TP Time Over Threshold Histogram")
    plt.xlabel("Time Over Threshold (Ticks)")

    plt.tight_layout()
    plt.savefig("tp_time_over_threshold_histogram.svg")
    plt.close()

def plot_flag_histogram(flags, quiet=False):
    """
    Plot the TP flags histogram.

    Generally, something notable. Likely difficult to read as a histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(flags, color='k')

    plt.title("TP Flags Histogram")
    plt.xlabel("Flags")

    plt.tight_layout()
    plt.savefig("tp_flags_histogram.svg")
    plt.close()

def plot_detid_histogram(det_ids, quiet=False):
    """
    Plot the TP detector IDs histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(det_ids, color='k')
    # Uncertain what the values of the det_ids refer to, so they remain as numbers.

    plt.title("TP Detector IDs Histogram")
    plt.xlabel("Detector IDs")

    plt.tight_layout()
    plt.savefig("tp_det_ids_histogram.svg")
    plt.close()

def plot_algorithm_histogram(algorithms, quiet=False):
    """
    Plot the TP algoritm histogram.

    Generally, this should be a single histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(algorithms, bins=np.arange(-0.5, 2, 1), range=(0,1), align='mid', color='k')

    plt.title("TP Algorithms Histogram")
    plt.xlabel("Algorithm")
    plt.xticks([0,1], ["Unknown", "TPCDefault"])
    # Values taken from TriggerPrimitive.hpp

    plt.tight_layout()
    plt.savefig("tp_algorithms_histogram.svg")
    plt.close()

def plot_adc_peak_histogram(adc_peaks, quiet=False):
    """
    Plot the TP peak histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(adc_peaks, color='k')

    plt.title("TP ADC Peaks Histogram")
    plt.xlabel("ADC Count")

    plt.tight_layout()
    plt.savefig("tp_adc_peaks_histogram.svg")
    plt.close()

def plot_adc_integral_histogram(adc_integrals, quiet=False):
    """
    Plot the TP ADC integral histogram.
    """
    plt.figure(figsize=(6,4))

    plt.hist(adc_integrals, color='k')

    plt.title("TP ADC Integrals Histogram")
    plt.xlabel("ADC Integral")

    plt.tight_layout()
    plt.savefig("tp_adc_integral_histogram.svg")
    plt.close()

def write_summary_stats(data, filename, title):
    """
    Writes the given summary statistics to 'filename'.
    """
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
        if os.path.isfile(anomaly_filename):
            # Prepare a new tp_anomaly_summary.txt
            os.remove(anomaly_filename)

    with PdfPages("tp_summary_stats.pdf") as pdf:
        for tp_key, title in titles.items():
            plt.figure(figsize=(6,4))
            ax = plt.gca()

            # Only plot the 'tp_key'
            plt.boxplot(tp_data[tp_key], notch=True, vert=False, sym='+')
            plt.yticks([])
            ax.xaxis.grid(True)
            plt.xlabel(labels[tp_key])

            plt.title(title)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Write anomalies to file.
            if not no_anomaly:
                if "Sanity" in title and np.all(tp_data[tp_key] == tp_data[tp_key][0]):
                    # Either passed check or all wrong in the same way.
                    continue
                write_summary_stats(tp_data[tp_key], anomaly_filename, title)

def parse():
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")
    parser.add_argument("--quiet", action="store_true", help="Stops the output of printed information. Default: False.")
    parser.add_argument("--start-frag", type=int, help="Fragment to start loading from (inclusive); can take negative integers. Default: -10", default=-10)
    parser.add_argument("--end-frag", type=int, help="Fragment to stop loading at (exclusive); can take negative integers. Default: 0", default=0)
    parser.add_argument("--no-anomaly", action="store_true", help="Pass to not write 'tp_anomaly_summary.txt'. Default: False.")

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
        print("Size of tp_data:", data.tp_data.shape)

    ## Plots with more involved analysis
    channel_tot(data.tp_data)
    tp_percent_histogram(data.tp_data)

    ## Basic Plots: Histograms & Box Plots
    # For the moment, none of these functions make use of 'quiet'.
    plot_adc_integral_histogram(data.tp_data['adc_integral'], quiet)
    plot_adc_peak_histogram(data.tp_data['adc_peak'], quiet)
    plot_algorithm_histogram(data.tp_data['algorithm'], quiet)
    plot_channel_histogram(data.tp_data['channel'], quiet)
    plot_detid_histogram(data.tp_data['detid'], quiet)
    plot_flag_histogram(data.tp_data['flag'], quiet)
    plot_time_over_threshold_histogram(data.tp_data['time_over_threshold'], quiet)
    plot_time_peak_histogram(data.tp_data['time_peak'], quiet)
    plot_time_start_histogram(data.tp_data['time_start'], quiet)
    plot_type_histogram(data.tp_data['type'], quiet)
    plot_version_histogram(data.tp_data['version'], quiet)

    plot_summary_stats(data.tp_data, no_anomaly, quiet)

if __name__ == "__main__":
    main()
