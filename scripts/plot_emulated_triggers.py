#!/usr/bin/env python

"""
Plot the trigger candidates, with their trigger activities & trigger primitives
for a given tpstream file
"""

import trgtools

from typing import Any
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtp
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import argparse

def parse() -> dict[str, Any]:
    """
    Parses CLI input arguments.

    Returns:
        (dict[str, Any]): A dictionary of argument names vs argument values
    """
    parser = argparse.ArgumentParser(
        description="Display diagnostic information for TCs for a given HDF5 file."
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
        help="Starting fragment index to process from. Takes negative indexing (Default: -10), NOT SUPPORTED YET).",
        default=-10
    )
    parser.add_argument(
        "--end-frag",
        type=int,
        help="Fragment index to stop processing (i.e. not inclusive). Takes negative indexing (Default: N(0), NOT SUPPORTED YET).",
        default=0
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Do you want to run in batch mode (e.g. without loading bars/tqdm)?"
    )

    return parser.parse_args()

def plot_all_event_displays(tc_data: list[NDArray],
                            tc_data_tas: list[NDArray],
                            ta_data: list[NDArray],
                            ta_data_tps: list[NDArray],
                            run_id: int,
                            file_index: int,
                            batch: bool) -> None:
    """

    Plots all the event displays, one per TC.

    Each event display will contain TriggerActivities (red boxes), and
    TriggerPrimitives (black points) for each TriggerActivity.

    Args:
        tc_data (list[NDArray]): A list of TriggerCandidates
        tc_data_tas (list[NDArray]): A list of TriggerActivityData per TriggerCandidate
        ta_data (list[NDArray]): A full list of TriggerActivities
        ta_data_tps (list[NDArray]): A list of TriggerPrimitives per TriggerActivity
        run_id (int): Run ID number
        file_index (int): File index
    """

    time_unit = "Ticks"

    with PdfPages(f"event_displays_{run_id}.{file_index:04}.pdf") as pdf:
        for tcdx, (tc, tas) in tqdm(enumerate(zip(tc_data, tc_data_tas)), total=len(tc_data), desc="Saving event displays", disable=batch):
            plt.figure(figsize=(6, 4))

            yend = tc["time_end"] - tc["time_start"]
            ta_times_starts = tas['time_start'] - tc["time_start"]
            ta_times_ends = tas['time_end'] - tc["time_start"] 

            channel_start = np.min(tas['channel_start'])
            channel_end = np.max(tas['channel_end'])

            # Only change the xlim if there is more than one TP.
            yexpansion = yend * 0.05
            xexpansion = (channel_end - channel_start) * 0.05

            currentAxis = plt.gca()
            for tadx, ta in enumerate(tas):
                rectangle = mtp.patches.Rectangle((ta['channel_start'], ta_times_starts[tadx]), ta['channel_end'] - ta['channel_start'],  ta_times_ends[tadx] - ta_times_starts[tadx], linewidth=1, edgecolor='r', facecolor='none')
                currentAxis.add_patch(rectangle)

                for tatmpdx, tatmp in enumerate(ta_data):
                    if (tatmp['time_start'] == ta['time_start']) and (tatmp['time_end'] == ta['time_end']) and (tatmp['channel_start'] == ta['channel_start']) and (tatmp['channel_end'] == ta['channel_end']):
                        time_starts = ta_data_tps[tatmpdx]['time_start'] - tc["time_start"]
                        plt.scatter(ta_data_tps[tatmpdx]['channel'], time_starts, lw=0, color='black', marker=',', s=1)

            plt.title(f'Run {run_id}.{file_index:04} Event Display: {tcdx:03}')
            plt.ylabel(f"Relative Start Time ({time_unit})")
            plt.xlabel("Channel")
            plt.ylim(0 - yexpansion, yend + yexpansion)
            plt.xlim(channel_start - xexpansion, channel_end + xexpansion)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

def main():
    args = parse()
    filename = args.filename
    verbosity = args.verbose
    #start_frag = args.start_frag
    #end_frag = args.end_frag
    batch = args.batch

    # Getting the ta data
    ta_reader = trgtools.TAReader(filename, verbosity, batch)
    ta_reader.read_all_fragments()

    # Getting the tc data
    tc_reader = trgtools.TCReader(filename, verbosity, batch)
    tc_reader.read_all_fragments()

    # Make the displays
    plot_all_event_displays(tc_reader.tc_data, tc_reader.ta_data, ta_reader.ta_data, ta_reader.tp_data, tc_reader.run_id, tc_reader.file_index, batch)

    print(f"From TCReader: number of TCs: {len(tc_reader.tc_data)}, all the TAs in TCs: {len(np.concatenate(tc_reader.ta_data))}")
    print(f"From TAReader: number of TAs: {len(ta_reader.tp_data)}")

if __name__ == "__main__":
    main()