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

def plot_adc_integral_vs_peak(tp_data):
    """
    Plot the ADC Integral vs ADC Peak.
    """
    plt.figure(figsize=(6, 4), dpi=200)
    plt.scatter(tp_data['adc_peak'], tp_data['adc_integral'], c='k', s=2, label='TP')
    #plt.plot(np.linspace(np.min(tp_data['adc_peak']), np.max(tp_data['adc_peak'])), color='#EE442F', label='Reference')
    plt.hlines(np.power(2, 15), np.min(tp_data['adc_peak']), np.max(tp_data['adc_peak']), color='#EE442F', label=r'$2^{15}-1$', alpha=0.2)
    print("Number of ADC Integrals at Signed 16 Limit:", np.sum(tp_data['adc_integral'] == np.power(2, 15)-1))
    print("Total number of TPs:", len(tp_data['adc_peak']))
    high_integral_locs = np.where(tp_data['adc_integral'] == np.power(2, 15)-1)
    plt.scatter(tp_data['adc_peak'][high_integral_locs], tp_data['adc_integral'][high_integral_locs], c='#63ACBE', s=2, marker='+', label=r'$2^{15}-1$')

    plt.title("ADC Integral vs ADC Peak")
    plt.xlabel("ADC Peak")
    plt.ylabel("ADC Integral")
    plt.legend()

    plt.tight_layout()
    plt.savefig("tp_adc_integral_vs_peak.png")  # Many scatter plot points makes this a PNG
    plt.close()

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
        bins = len(plot_details_dict['xticks'][1])
        plt.xticks(
                plot_details_dict['xticks'][0],  # Ticks to change
                plot_details_dict['xticks'][1]  # New labels
        )

    if linear and log:
        ax.hist(data, bins=bins, color='#EE442F', label='Linear', alpha=0.6)
        ax.set_yscale('linear')

        ax2 = ax.twinx()
        ax2.hist(data, bins=bins, color='#63ACBE', label='Log', alpha=0.6)
        ax2.set_yscale('log')

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
    plt.xlabel(plot_details_dict['xlabel'])
    if 'xticks' in plot_details_dict:
        plt.xticks(plot_details_dict['xticks'][0], plot_details_dict['xticks'][1])
    if 'xlim' in plot_details_dict:
        plt.xlim(plot_details_dict['xlim'])

    plt.tight_layout()
    pdf.savefig()
    plt.close()

def write_summary_stats(data, filename, title):
    """
    Writes the given summary statistics to 'filename'.
    """
    # Algorithm, Det ID, etc. are not expected to vary.
    # Check first that they don't vary, and move on if so.
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all TPs. Skipping summary statistics.")
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
    parser.add_argument("--linear", action="store_true", help="Pass to use linear histogram scaling. Default: plots both linear and log.")
    parser.add_argument("--log", action="store_true", help="Pass to use logarithmic histogram scaling. Default: plots both linear and log.")

    return parser.parse_args()

def main():
    """
    Drives the processing and plotting.
    """
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet
    start_frag = args.start_frag
    end_frag = args.end_frag
    no_anomaly = args.no_anomaly
    linear = args.linear
    log = args.log

    # User didn't pass either flag, so default to both being true.
    if (not linear) and (not log):
        linear = True
        log = True

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
    plot_adc_integral_vs_peak(data.tp_data)

    ## Basic Plots: Histograms & Box Plots
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
                'xlim': (-0.5, 1.5),
                'xticks': (
                    (0, 1),  # xticks to change
                    ("Unknown", "TPCDefault"),
                )
            },
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
                'xlabel': "Ticks"
            },
            'time_peak': {
                'title': "Relative Time Peak",
                'xlabel': "Ticks"
            },
            'time_start': {
                'title': "Relative Time Start",
                'xlabel': "Ticks"
            },
            'type': {
                'title': "Type",
                'xlabel': "Type",
                'xlim': (-0.5, 2.5),
                'xticks': (
                    (0, 1, 2),  # Ticks to change
                    ('Unknown', 'TPC', 'PDS')
                )
            },
            'version': {
                'title': "Version",
                'xlabel': "Versions"
            }
    }
    if not no_anomaly:
        anomaly_filename = "tp_anomalies.txt"
        if not quiet:
            print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new tp_anomaly_summary.txt
            os.remove(anomaly_filename)

    with PdfPages("tp_data_member_histograms.pdf") as pdf:
        for tp_key in data.tp_data.dtype.names:
            if tp_key == 'time_start' or tp_key == 'time_peak':
                min_time = np.min(data.tp_data[tp_key])
                plot_pdf_histogram(data.tp_data[tp_key] - min_time, plot_dict[tp_key], pdf, linear, log)
                continue
            plot_pdf_histogram(data.tp_data[tp_key], plot_dict[tp_key], pdf, linear, log)
            if not no_anomaly:
                write_summary_stats(data.tp_data[tp_key], anomaly_filename, plot_dict[tp_key]['title'])

if __name__ == "__main__":
    main()
