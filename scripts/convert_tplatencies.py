#!/usr/bin/env python
"""
Converts the large csv TA latencies output from the trigger emulator to a
simpler format per-ta that can be used for plotting. Also requires the raw hdf5
emulation output to find and parse all the TAs.
"""
import trgtools

import pandas as pd
import numpy as np

import argparse
import os

def parse():
    """
    Parses CLI input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert full csv latency output from emulator to simpler format for plotting."
    )
    parser.add_argument(
        "--verbose", '-v',
        action="count",
        help="Increment the verbose level (errors, warnings, all)."
        "Save names and skipped writes are always printed. Default: 0.",
        default=0
    )
    parser.add_argument(
        "--latencies", '-l',
        help=".csv file with the full latency output"
    )
    parser.add_argument(
        "--raw", '-r',
        help=".hdf5 raw emulation output with TPs and TAs"
    )
    return parser.parse_args()

def main():
    # Get the input arguments
    args = parse()
    verbosity = args.verbose
    input_latencies = args.latencies
    input_raw = args.raw

    # Load the HDF5 & CSV data
    print("Loading TAs from hdf5");
    ta_data = trgtools.TAReader(input_raw, verbosity)
    ta_data.read_all_fragments()
    print("Loading TPs from csv");
    tp_data_latencies = pd.read_csv(input_latencies)

    # Pre-allocate the output latencies
    output_array = np.zeros((len(ta_data.tp_data), 5), dtype=np.uint64)

    # Find all the TPs that caused creation of a TA
    tpidx_making_tas = np.where(tp_data_latencies.iloc[:, 3].values == 1)

    last_tpidx_making_tas = 0

    # Iterate over all the TAs
    for taidx, ta in enumerate(ta_data.tp_data):
        # Find the first / last TP in a TA
        first_tp_time   = ta_data.ta_data[taidx]["time_start"]
        last_tp_time    = ta_data.ta_data[taidx]["time_end"]
        first_tp_idx    = np.searchsorted(tp_data_latencies.iloc[:, 0], first_tp_time)
        last_tp_idx     = np.searchsorted(tp_data_latencies.iloc[:, 0], last_tp_time, side="right")

        # Create TP data window based on the TA window to calculate latencies from
        window_df = tp_data_latencies.iloc[first_tp_idx:last_tp_idx + 1]
        latencies_tawindow_sum  = window_df.iloc[:, 2].sum()
        latencies_tawindow_mean = window_df.iloc[:, 2].mean()

        # Add up latencies per tp that's inside of the TA.
        # TODO: I did not implement this because it's too slow
        #latencies_tasum = 0
        #for tp_ta in ta:
        #    for idx, (tp_window_time, tp_window_adc) in enumerate(zip(window_df.iloc[:,0], window_df.iloc[:,1])):
        #        if tp_ta[8] == tp_window_time and tp_ta[0] == tp_window_adc:
        #            latencies_tasum += window_df.iloc[idx][3]

        # Find the next TP that made a TA
        tp_making_ta_idx = None
        for idx, tpidx in enumerate(tpidx_making_tas[0][last_tpidx_making_tas:]):
            if tp_data_latencies.iloc[tpidx, 0] > first_tp_time:
                tp_making_ta_idx = tpidx
                # Update the last tp index that made a TA for faster enumeration above
                if (idx + last_tpidx_making_tas) > last_tpidx_making_tas:
                    last_tpidx_making_tas = idx + last_tpidx_making_tas
                break

        # Special case when the last TP is not found. Skip this TA.
        if tp_making_ta_idx == None: continue
        latencies_whole = tp_data_latencies.iloc[first_tp_idx + 1:tp_making_ta_idx + 1,2].sum()
        latencies_lasttp = tp_data_latencies.iloc[tp_making_ta_idx,2]

        output_array[taidx][0] = latencies_lasttp
        output_array[taidx][1] = latencies_whole
        output_array[taidx][2] = latencies_tawindow_sum
        output_array[taidx][3] = latencies_tawindow_mean
        output_array[taidx][4] = len(ta)

    df = pd.DataFrame(output_array)
    output_dir, output_filename = os.path.split(input_latencies)
    output_file = os.path.join(output_dir, os.path.splitext(output_filename)[0] + '_simplified.csv')

    print(f'Saving the output in csv file: {output_file}')
    df.to_csv(output_file,index=False, header=False)

if __name__ == "__main__":
    main()
