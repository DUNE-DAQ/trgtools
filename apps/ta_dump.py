"""
Display diagnostic information for TAs for a given
tpstream file.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import trgtools

def window_length_hist(window_lengths):
    """
    Plot a histogram of the TA window lengths.
    """
    plt.figure(figsize=(6,4))
    plt.hist(np.array(window_lengths), color='k')

    plt.title("TA Time Window Length Histogram")
    plt.xlabel("Time Window Length")

    plt.savefig("window_length_histogram.svg")
    plt.close()

def num_tps_hist(num_tps):
    """
    Plot the number of TPs for each TA as a histogram.
    """
    plt.figure(figsize=(6,4))
    plt.hist(np.array(num_tps), color='k')

    plt.title("Number of TPs Histogram")
    plt.xlabel("Number of TPs")

    plt.savefig("num_tps_histogram.svg")
    plt.close()

def time_start_plot(start_times):
    """
    Plot in order the time_start member data for TAs.
    """
    plt.figure(figsize=(6,4))
    plt.plot(start_times, 'ok')

    plt.title("TA Start Times")
    plt.xlabel("TA Order")
    plt.ylabel("Start Time")

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
                                         "PlaceCoincidence"), rotation=60)

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

def parse():
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")

    return parser.parse_args()

def main():
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    diagnostics = {
                    "num_tps": [],
                    "window_length": [],
                    "algorithm": [],
                    "det_type": [],
                    "adc_integral": [],
                    "time_start": []
                  }

    data = trgtools.TAData(filename)
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

    num_tps_hist(diagnostics["num_tps"])
    window_length_hist(diagnostics["window_length"])
    algorithm_hist(diagnostics["algorithm"])
    det_type_hist(diagnostics["det_type"])
    adc_integral_hist(diagnostics["adc_integral"])
    time_start_plot(diagnostics["time_start"])

if __name__ == "__main__":
    main()
