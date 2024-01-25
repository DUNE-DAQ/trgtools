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
    plt.hist(np.array(window_lengths, dtype=np.uint64).flatten() * TICK_TO_SEC_SCALE, color='k')

    plt.title("TA Time Window Length Histogram")
    plt.xlabel("Time Window Length (s)")

    plt.savefig("window_length_histogram.svg")
    plt.close()

def num_tps_hist(num_tps):
    """
    Plot the number of TPs for each TA as a histogram.
    """
    plt.figure(figsize=(6,4))
    plt.hist(np.array(num_tps).flatten(), bins=50, color='k')

    plt.title("Number of TPs Histogram")
    plt.xlabel("Number of TPs")

    plt.savefig("num_tps_histogram.svg")
    plt.close()

def time_start_plot(start_times, frag_index=-1):
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
    """
    Plot a histogram of the algorithm types for each TA.
    """
    plt.figure(figsize=(12,8))
    counts, _ , _ = plt.hist(np.array(algorithms).flatten(), bins=np.arange(-0.5, 8, 1), range=(0,7), align='mid', color='k')
    print(f"Number of TAs: {np.sum(counts).astype(int)}")

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
    """
    Plot a histogram of the detector type for the TAs.
    """
    plt.figure(figsize=(12,8))
    plt.hist(np.array(det_types).flatten(), bins=np.arange(-0.5, 3, 1), range=(0,2), align='mid', color='k')

    plt.title("TA Detector Type Histogram")
    plt.xticks(ticks=range(0,3), labels=("Unknown",
                                         "TPC",
                                         "PDS"), rotation=60)

    plt.savefig("det_types_histogram.svg")
    plt.close()

def adc_integral_hist(adc_integrals):
    """
    Plot a histogram of the ADC integrals for the TAs.
    """
    plt.figure(figsize=(6,4))
    plt.hist(np.array(adc_integrals).flatten(), color='k')

    plt.title("TA ADC Integral Histogram")
    plt.xlabel("ADC Integral")

    plt.savefig("adc_integral_histogram.svg")
    plt.close()

def all_event_displays(tp_data, run_id, sub_run_id):
    """
    Plot all event_displays as pages in a PDF.
    """
    with PdfPages(f"event_displays_{run_id}.{sub_run_id:04}.pdf") as pdf:
        for fdx, frag in enumerate(tp_data):
            for tadx, ta in enumerate(frag):
                fig = plt.figure(figsize=(6,4))

                plt.scatter(ta['time_peak'], ta['channel'], c='k', s=2)

                max_time = np.max(ta['time_peak'])
                min_time = np.min(ta['time_peak'])
                time_diff = max_time - min_time

                plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))
                plt.title(f'Run {run_id}.{sub_run_id:04} Event Display: {fdx:03}.{tadx:03}')
                plt.xlabel("Peak Time")
                plt.ylabel("Channel")

                pdf.savefig()
                plt.close()

def time_diff_hist(start_times, end_times):
    """
    Plot a histogram of the time differences.
    """
    # Difference between all the start times.
    start_time_diff = (np.concatenate((start_times[1:], [0])) - np.array(start_times))[:-1]
    # Difference between previous TA end time and current TA start time.
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
    """
    Plot an individual event display.

    """
    plt.figure(figsize=(6,4))

    plt.scatter(peak_times, channels, c='k', s=2)
    max_time = np.max(peak_times)
    min_time = np.min(peak_times)
    time_diff = max_time - min_time

    plt.xlim((min_time - 0.1*time_diff, max_time + 0.1*time_diff))

    plt.title(f"Event Display: {idx:03}")
    plt.xlabel("Peak Time")
    plt.ylabel("Channel")

    plt.savefig(f"./event_display_{idx:03}.svg")
    plt.close()

def parse():
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")
    parser.add_argument("--quiet", action="store_true", help="Stops the output of printed information. Default: False.")
    parser.add_argument("--no-displays", action="store_true", help="Stops the processing of event displays.")
    parser.add_argument("--start-frag", type=int, help="Starting fragment index to process from. Takes negative indexing. Default: -10.", default=-10)
    parser.add_argument("--end-frag", type=int, help="Fragment index to stop processing (i.e. not inclusive). Takes negative indexing. Default: 0.", default=0)

    return parser.parse_args()

def main():
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet
    no_displays = args.no_displays
    start_frag = args.start_frag
    end_frag = args.end_frag

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
    if end_frag != 0:
        frag_paths = data.get_ta_frag_paths()[start_frag:end_frag]
    elif end_frag == 0:
        frag_paths = data.get_ta_frag_paths()[start_frag:]
    for path in frag_paths:
        data.load_frag(path)

    run_id = data.run_id
    sub_run_id = data.sub_run_id

    for frag in data.ta_data:
        for ta in frag:
            diagnostics["num_tps"].append(ta["num_tps"])
            diagnostics["window_length"].append(ta["time_end"] - ta["time_start"])
            diagnostics["algorithm"].append(ta["algorithm"])
            diagnostics["det_type"].append(ta["type"])
            diagnostics["adc_integral"].append(ta["adc_integral"])
            diagnostics["time_start"].append(ta["time_start"])
            diagnostics["time_end"].append(ta["time_end"])

    if (not quiet):
        print(f"Number of Fragments: {len(data.ta_data)}")

    ## Plotting
    num_tps_hist(diagnostics["num_tps"])
    window_length_hist(diagnostics["window_length"])
    algorithm_hist(diagnostics["algorithm"])
    det_type_hist(diagnostics["det_type"])
    adc_integral_hist(diagnostics["adc_integral"])
    time_start_plot(np.array(diagnostics["time_start"]).flatten())
    time_diff_hist(np.array(diagnostics["time_start"]).flatten(), np.array(diagnostics["time_end"]).flatten())

    if (not no_displays):
        all_event_displays(data.tp_data, run_id, sub_run_id)

if __name__ == "__main__":
    main()
