#!/usr/bin/env python3
"""
Plot an arbitrary part of a TimeSlice from a TPStream.
"""

from trgtools import TPReader

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pathlib import Path

from rich import print


def plot_tps_channel_vs_time(tps: npt.NDArray) -> None:
    times: npt.NDArray[int] = tps['time_start'] - tps['time_start'].min()
    channels: npt.NDArray[int] = tps['channel']

    plt.figure(figsize=(6, 4), dpi=600, layout="constrained")
    plt.box(False)

    plt.scatter(times, channels, c='k', s=0.1, alpha=0.8)

    plt.xlabel("Time (16 ns / tick)")
    plt.ylabel("Channel Number")
    plt.title("Example TP Display")

    plt.savefig("example_tp_display.pdf")
    plt.close
    return



@click.command()
@click.argument("file_path", type=click.Path(readable=True, path_type=Path))
def main(file_path: Path) -> int:
    tpr: TPReader = TPReader(str(file_path.expanduser()), verbosity=2)

    print(tpr.get_fragment_paths())

    # Get a random fragment
    fragment_path: str = np.random.choice(tpr.get_fragment_paths())
    print(fragment_path)
    tps: npt.NDArray = tpr.read_fragment(fragment_path)

    # # Get a random TP to base from (T0)
    # tp: npt.NDArray = np.random.choice(tps)
    # time_extension: float = 0.005 / 16e-9  # 5 ms / (16 ns per tick)
    # time_start: int = tp['time_start']
    # time_end: int = int(tp['time_start'] + time_extension)

    # mask: npt.NDArray[bool] = (tps['time_start'] >= time_start) & (tps['time_start'] < time_end)
    # tps = tps[mask]

    plot_tps_channel_vs_time(tps)
    return 0


if __name__ == "__main__":
    main()
