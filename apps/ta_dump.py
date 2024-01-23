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

import trgtools

TICK_TO_SEC_SCALE = 512e-9 # s per tick

def window_length_hist(window_lengths):
    """
    Plot a histogram of the TA window lengths.
    """
    plt.figure(figsize=(6,4))
    plt.hist(np.array(window_lengths, dtype=np.uint64) * TICK_TO_SEC_SCALE, color='k')

    plt.title("TA Time Window Length Histogram")
    plt.xlabel("Time Window Length (s)")

    plt.savefig("window_length_histogram.svg")
    plt.close()

def num_tps_hist(num_tps):
    """
    Plot the number of TPs for each TA as a histogram.
    """
    plt.figure(figsize=(6,4))
    plt.hist(num_tps, bins=50, color='k')

    plt.title("Number of TPs Histogram")
    plt.xlabel("Number of TPs")
    #plt.xlim((0,500))
    #plt.ylim((0,200))

    plt.savefig("num_tps_histogram.svg")
    plt.close()

def time_start_plot(start_times):
    """
    Plot in order the time_start member data for TAs.
    """
    first_time = start_times[0] * TICK_TO_SEC_SCALE
    total_ticks = start_times[-1] - start_times[0]
    total_tas = len(start_times)
    avg_rate_ticks = total_tas / total_ticks
    avg_rate_secs = total_tas / (total_ticks * TICK_TO_SEC_SCALE)

    plt.figure(figsize=(6,4))
    plt.plot(np.array(start_times, dtype=np.uint64) * TICK_TO_SEC_SCALE - first_time, 'k', label=f"Average Rate: {avg_rate_secs:.3} TA/s")

    plt.title(f"TA Start Times: Shifted by {first_time:.4e} s")
    plt.xlabel("TA Order")
    plt.ylabel("Start Time (s)")

    plt.legend()

    plt.savefig("start_times.svg")
    plt.close()

def algorithm_hist(algorithms):
    plt.figure(figsize=(12,8))
    plt.hist(np.array(algorithms), bins=np.arange(-0.5, 8, 1), range=(0,7), align='mid', color='k')

    plt.title("TA Algorithm Histogram")
    plt.xticks(ticks=range(0,8), labels=("Unknown",
                                         "Supernova",
                                         "Prescale",
                                         "ADCSimpleWindow",
                                         "HorizontalMuon",
                                         "MichelElectron",
                                         "DBSCAN",
                                         "PlaneCoincidence"), rotation=60)

    plt.tight_layout()
    plt.savefig("algorithm_histogram.svg")
    plt.close()

def det_type_hist(det_types):
    plt.figure(figsize=(12,8))
    plt.hist(np.array(det_types), bins=np.arange(-0.5, 3, 1), range=(0,2), align='mid', color='k')

    plt.title("TA Detector Type Histogram")
    plt.xticks(ticks=range(0,3), labels=("Unknown",
                                         "TPC",
                                         "PDS"), rotation=60)

    plt.savefig("det_types_histogram.svg")
    plt.close()

def adc_integral_hist(adc_integrals):
    plt.figure(figsize=(6,4))
    plt.hist(np.array(adc_integrals), color='k')

    plt.title("TA ADC Integral Histogram")
    plt.xlabel("ADC Integral")

    plt.savefig("adc_integral_histogram.svg")
    plt.close()

def all_event_displays(tp_data, run_id, sub_run_id):
    """
    Plot all event_displays as pages in a PDF.
    """
    with PdfPages(f"event_displays_{run_id}.{sub_run_id:04}.pdf") as pdf:
        for idx, tp_datum in enumerate(tp_data):
            fig = plt.figure(figsize=(6,4))

            plt.scatter(tp_datum['time_peak'], tp_datum['channel'], c='k', s=2)

            max_time = np.max(tp_datum['time_peak'])
            min_time = np.min(tp_datum['time_peak'])
            time_diff = max_time - min_time

            plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))
            plt.title(f'Run {run_id}.{sub_run_id:04} Event Display: {idx:03}')
            plt.xlabel("Peak Time")
            plt.ylabel("Channel")

            pdf.savefig()
            plt.close()

def start_time_diff_hist(start_times):
    start_time_diff = (np.array(start_times[1:] + [0]) - np.array(start_times))[:-1]
    plt.figure(figsize=(6,4))

    plt.hist(start_time_diff, bins=25, color='#63ACBE', label="Start Time Difference", alpha=1.0)

    plt.title("TA Timings Histogram")
    plt.xlabel("Time Ticks")
    plt.legend()

    plt.savefig("start_time_diff_histogram.svg")
    plt.close()

def time_diff_hist(start_times, end_times):
    start_time_diff = (np.array(start_times[1:] + [0]) - np.array(start_times))[:-1]
    time_gaps = np.array(start_times)[1:] - np.array(end_times)[:-1]

    start_time_diff = start_time_diff.astype(np.uint64) * TICK_TO_SEC_SCALE
    time_gaps = time_gaps.astype(np.uint64) * TICK_TO_SEC_SCALE
    plt.figure(figsize=(6,4))

    plt.hist(start_time_diff, bins=40, color='#63ACBE', label="Start Time Difference", alpha=0.2)
    plt.hist(time_gaps, bins=40, color="#EE442F", label="TA Time Gap", alpha=0.2)

    plt.title("TA Timings Histogram")
    plt.xlabel("Seconds")
    plt.legend()

    plt.savefig("ta_timings_histogram.svg")
    plt.close()

def event_display(peak_times, channels, idx):
    plt.figure(figsize=(6,4))

    plt.scatter(peak_times, channels, c='k', s=2)
    max_time = np.max(peak_times)
    min_time = np.min(peak_times)
    time_diff = max_time - min_time

    plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))

    plt.title(f"Event Display: {idx:03}")
    plt.xlabel("Peak Time")
    plt.ylabel("Channel")

    plt.savefig(f"./event_displays/event_display_{idx:03}.svg")
    plt.close()

def parse():
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")
    parser.add_argument("--quiet", action="store_true", help="Stops the output of printed information. Default: False.")
    parser.add_argument("--no-displays", action="store_true", help="Stops the processing of event displays.")

    return parser.parse_args()

def main():
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet
    no_displays = args.no_displays

    diagnostics = {
                    "num_tps": [],
                    "window_length": [],
                    "algorithm": [],
                    "det_type": [],
                    "adc_integral": [],
                    "time_start": [],
                    "time_end": []
                  }

    data = trgtools.TAData(filename, quiet)
    data.load_all_frags()
    run_id = data.run_id
    sub_run_id = data.sub_run_id

    for ta in data.ta_data:
        diagnostics["num_tps"].append(ta["num_tps"])
        diagnostics["window_length"].append(ta["time_end"] - ta["time_start"])
        diagnostics["algorithm"].append(ta["algorithm"])
        diagnostics["det_type"].append(ta["type"])
        diagnostics["adc_integral"].append(ta["adc_integral"])
        diagnostics["time_start"].append(ta["time_start"])
        diagnostics["time_end"].append(ta["time_end"])

    ## Plotting
    num_tps_hist(diagnostics["num_tps"])
    window_length_hist(diagnostics["window_length"])
    algorithm_hist(diagnostics["algorithm"])
    det_type_hist(diagnostics["det_type"])
    adc_integral_hist(diagnostics["adc_integral"])
    time_start_plot(diagnostics["time_start"])
    time_diff_hist(diagnostics["time_start"], diagnostics["time_end"])
    start_time_diff_hist(diagnostics["time_start"])

    if (not no_displays):
        all_event_displays(data.tp_data, run_id, sub_run_id)

if __name__ == "__main__":
    main()
