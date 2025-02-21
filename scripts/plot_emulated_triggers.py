#!/usr/bin/env python

"""
"""

import trgtools
from trgtools.plot import PDFPlotter

import trgdataformats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtp
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from tqdm import tqdm

import argparse
import os


ALGORITHM_LABELS = list(trgdataformats.TriggerCandidateData.Algorithm.__members__.keys())
ALGORITHM_TICKS = [tp_alg.value for tp_alg in trgdataformats.TriggerCandidateData.Algorithm.__members__.values()]
TYPE_LABELS = list(trgdataformats.TriggerCandidateData.Type.__members__.keys())
TYPE_TICKS = [tp_type.value for tp_type in trgdataformats.TriggerCandidateData.Type.__members__.values()]

TICK_TO_SEC_SCALE = 16e-9  # s per tick

def parse():
    """
    Parses CLI input arguments.
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
        help="Starting fragment index to process from. Takes negative indexing. Default: -10.",
        default=-10
    )
    parser.add_argument(
        "--end-frag",
        type=int,
        help="Fragment index to stop processing (i.e. not inclusive). Takes negative indexing. Default: N.",
        default=0
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        help="Do you want to overwrite the output plot file, if already exists? Default: False.",
        default=False
    )

    return parser.parse_args()

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
    save_name = f"tc_{run_id}-{file_index:04}_figures_{name_iter:04}"

    # Outputs will always create a PDF, so use that as the comparison.
    while not overwrite and os.path.exists(save_name + ".pdf"):
        name_iter += 1
        save_name = f"tc_{run_id}-{file_index:04}_figures_{name_iter:04}"
    print(f"Saving outputs to ./{save_name}.*")

    return save_name

def plot_all_event_displays(tc_data: list[np.ndarray],
                            tc_data_tas: list[np.ndarray],
                            ta_data: list[np.ndarray],
                            ta_data_tps: list[np.ndarray],
                            run_id: int,
                            file_index: int) -> None:

    time_unit = "Ticks"

    print("tick 0")
    with PdfPages(f"event_displays_{run_id}.{file_index:04}.pdf") as pdf:
        for tcdx, (tc, tas) in tqdm(enumerate(zip(tc_data, tc_data_tas)), total=len(tc_data), desc="Saving event displays"):
            plt.figure(figsize=(6, 4))

            yend = tc["time_end"] - tc["time_start"]
            ta_times_starts = tas['time_start'] - tc["time_start"]
            ta_times_ends = tas['time_end'] - tc["time_start"] 

            channel_start = np.min(tas['channel_start'])
            channel_end = np.max(tas['channel_end'])

            # Only change the xlim if there is more than one TP.
            yexpansion = yend * 0.05
            xexpansion = (channel_end - channel_start) * 0.05

            plt.ylim(0 - xexpansion, yend + xexpansion)
            plt.xlim(channel_start - yexpansion, channel_end + yexpansion)

            currentAxis = plt.gca()
            for tadx, ta in enumerate(tas):
                rectangle = mtp.patches.Rectangle((ta['channel_start'], ta_times_starts[tadx]), ta['channel_end'] - ta['channel_start'],  ta_times_ends[tadx] - ta_times_starts[tadx], linewidth=1, edgecolor='r', facecolor='none')
                currentAxis.add_patch(rectangle)

                for tatmpdx, tatmp in enumerate(ta_data):
                    if (tatmp['time_start'] == ta['time_start']) and (tatmp['time_end'] == ta['time_end']) and (tatmp['channel_start'] == ta['channel_start']) and (tatmp['channel_end'] == ta['channel_end']):
                        time_starts = ta_data_tps[tatmpdx]['time_start'] - tc["time_start"]
                        plt.scatter(ta_data_tps[tatmpdx]['channel'], time_starts, lw=0, color='black', marker=',', s=1)

            plt.title(f'Run {run_id}.{file_index:04} Event Display: {tadx:03}')
            plt.ylabel(f"Relative Start Time ({time_unit})")
            plt.xlabel("Channel")
            plt.ylim(0 - xexpansion, yend + xexpansion)
            plt.xlim(channel_start - yexpansion, channel_end + yexpansion)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

def main():
    args = parse()
    filename = args.filename
    verbosity = args.verbose
    #start_frag = args.start_frag
    #end_frag = args.end_frag
    overwrite = args.overwrite

    # Getting the ta data
    ta_reader = trgtools.TAReader(filename, verbosity)
    ta_reader.read_all_fragments()

    # Getting the tc data
    tc_reader = trgtools.TCReader(filename, verbosity)
    tc_reader.read_all_fragments()

    # Create the output file
    save_name = find_save_name(tc_reader.run_id, tc_reader.file_index, overwrite)
    pdf_plotter = PDFPlotter(f"{save_name}.pdf")
    pdf = pdf_plotter.get_pdf()  # Needed for extra plots that are not general.

    # Make the displays
    plot_all_event_displays(tc_reader.tc_data, tc_reader.ta_data, ta_reader.ta_data, ta_reader.tp_data, tc_reader.run_id, tc_reader.file_index)

    print(f"tc data: {tc_reader}: number of tcs: {len(tc_reader.tc_data)}, all the TAs in TCs: {len(tc_reader.ta_data)}")
    print(f"ta data: {ta_reader} number of TAs using tp data: {len(ta_reader.tp_data)}")# and ta data: {ta_reader.ta_data}")

if __name__ == "__main__":
    main()
