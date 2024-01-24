"""
Display diagnostic information for TPs in a given
tpstream file.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

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
    plt.ylabel("Time Over Threshold (s)")
    plt.legend()

    plt.annotate(f"High ToT Channels: {np.unique(hot_channels)}", xy=(2250, 0.0175), va="center", ha="center", fontsize=8)

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

    plt.savefig("percent_total.svg")
    plt.close()

def tp_channel_histogram(tp_data):
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
    print("High TP Count Channels:", np.where(counts >= 500))
    print("Percentage Counts:", np.where(counts >= 0.01*total_counts))

    plt.figure(figsize=(6,4))

    plt.stairs(counts, bins, fill=True, color='k', label=f'TP Count: {total_counts}')

    plt.title("TP Channel Histogram")
    plt.xlabel("Channel")
    plt.legend()

    plt.ylim((0, 800))

    plt.savefig("tp_channel_histogram.svg")
    plt.close()

def parse():
    parser = argparse.ArgumentParser(description="Display diagnostic information for TAs for a given tpstream file.")
    parser.add_argument("filename", help="Absolute path to tpstream file to display.")
    parser.add_argument("--quiet", action="store_true", help="Stops the output of printed information. Default: False.")

    return parser.parse_args()

def main():
    ## Process Arguments & Data
    args = parse()
    filename = args.filename
    quiet = args.quiet

    data = trgtools.TPData(filename, quiet)
    num_frags = 10
    frag_paths = data.get_tp_frag_paths()
    for path in frag_paths[-1*num_frags:]:
        print(path)
        data.load_frag(path)

    print("Length of tp_data:", len(data.tp_data))
    tp_channel_histogram(data.tp_data)
    channel_tot(data.tp_data)
    tp_percent_histogram(data.tp_data)

if __name__ == "__main__":
    main()
