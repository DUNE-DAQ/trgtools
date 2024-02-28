#!/usr/bin/env python
"""
Display diagnostic information for HSIs for a given
HDF5 file.
"""
import trgtools
from trgtools.plot import PDFPlotter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from copy import deepcopy
import os
import argparse


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
    save_name = f"hsi_{run_id}-{file_index:04}_figures_{name_iter:04}"

    # Outputs will always create a PDF, so use that as the comparison.
    while not overwrite and os.path.exists(save_name + ".pdf"):
        name_iter += 1
        save_name = f"hsi_{run_id}-{file_index:04}_figures_{name_iter:04}"
    print(f"Saving outputs to ./{save_name}.*")

    return save_name


def write_summary_stats(data: np.ndarray, filename: str, title: str) -> None:
    """
    Writes the given summary statistics to :filename:.

    Parameters:
        data (np.ndarray): Array of a HSI data member.
        filename (str): File to append outputs to.
        title (str): Title of the HSI data member.

    Appends statistics to the given file.
    """
    if np.all(data == data[0]):
        print(f"{title} data member is the same for all HSIs.")
        print("Only writing the value and counts.")
        with open(filename, 'a') as out:
            out.write(f"{title}\n"
                      f"Reference Statistics:\n"
                      f"\tTotal # HSIs = {len(data)},\n"
                      f"\tValue = {data[0]}\n")
            out.write("\n\n")
        return None

    summary = stats.describe(data)
    std = np.sqrt(summary.variance)
    with open(filename, 'a') as out:
        out.write(f"{title}\n")
        out.write(f"Reference Statistics:\n"
                  f"\tTotal # HSIs = {summary.nobs},\n"
                  f"\tMean = {summary.mean:.2f},\n"
                  f"\tStd = {std:.2f},\n"
                  f"\tMin = {summary.minmax[0]},\n"
                  f"\tMax = {summary.minmax[1]}.\n")
        std3_count = np.sum(data > summary.mean + 3*std) + np.sum(data < summary.mean - 3*std)
        std2_count = np.sum(data > summary.mean + 2*std) + np.sum(data < summary.mean - 2*std)
        out.write(f"Anomalies:\n"
                  f"\t# of >3 Sigma HSIs = {std3_count},\n"
                  f"\t# of >2 Sigma HSIs = {std2_count}.\n")
        out.write("\n\n")

    return None


def plot_pdf_bitmap(data: np.ndarray, pdf: PdfPages, plot_details_dict: dict) -> None:
    """
    Plot a bitmap onto the PdfPages object.

    Parameters:
        data (np.ndarray): Array with entries as the active bit positions.
        pdf (PdfPages): PdfPages object to append to.
        plot_details_dict (dict): Dictionanry with keys such as 'title', 'xlabel', etc.

    Returns:
        Nothing. Mutates :pdf: with the new plot
    """
    # Find the maximum bit position to limit array size
    max_bit_pos = -np.Inf
    for bit_positions in data:
        if len(bit_positions) == 0:
            continue
        max_bit_pos = np.max((np.max(bit_positions), max_bit_pos))

    # Bitmaps were empty or only went up to 0
    if max_bit_pos <= 0:
        max_bit_pos = 8  # Arbitrary number choice.

    # Fill in the array with 0 and 1
    bitmap = np.zeros((len(data), int(max_bit_pos)), dtype=int)
    for idx, bit_positions in enumerate(data):
        bitmap[idx, bit_positions] = 1

    plt.figure(figsize=plot_details_dict.get('figsize', (6, 4)))
    plt.imshow(bitmap, **plot_details_dict.get('imshow_style', {'cmap':"Greys"}))

    if plot_details_dict.get('colorbar', False):
        plt.colorbar(**plot_details_dict.get('colorbar_style', {}))

    if plot_details_dict.get('grid', False):
        plt.grid(**plot_details_dict.get('grid_style', {'visible': True}))

    plt.title(plot_details_dict['title'])
    plt.xlabel(plot_details_dict['xlabel'])
    plt.ylabel(plot_details_dict['ylabel'])

    plt.xlim(plot_details_dict.get('xlim', (max_bit_pos-0.5, -0.5)))

    if plot_details_dict.get('entries', False):
        colors = plot_details_dict.get('entries_color', ('w', 'k'))
        for idx in range(len(data)):
            for jdx in range(max_bit_pos):
                c = colors[0] if bitmap[idx, jdx] else colors[1]
                plt.text(jdx, idx, bitmap[idx, jdx], ha='center', va='center', color=c)

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    return None


def parse():
    """
    Parses CLI input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display diagnostic information for HSIs for a given HDF5 file."
    )
    parser.add_argument(
        "filename",
        help="Absolute path to tpstream file to display."
    )
    parser.add_argument(
        "--verbose", '-v',
        action="count",
        help="Increment the verbose level (errors, warnings, all)."
        "Save names and skipped writes are always printed. Default: 0.",
        default=0
    )
    parser.add_argument(
        "--start-frag",
        type=int,
        help="Starting fragment index to process from. Takes negative indexing. Default: 0.",
        default=0
    )
    parser.add_argument(
        "--end-frag",
        type=int,
        help="Fragment index to stop processing (i.e. not inclusive). Takes negative indexing. Default: 2.",
        default=2
    )
    parser.add_argument(
        "--no-anomaly",
        action="store_true",
        help="Pass to not write 'ta_anomaly_summary.txt'. Default: False."
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Pass to use linear histogram scaling. Default: plots both linear and log."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Pass to use logarithmic histogram scaling. Default: plots both linear and log."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite old outputs. Default: False."
    )

    return parser.parse_args()


def main():
    """
    Drives the processing and plotting.
    """
    # Process Arguments & Data
    args = parse()
    filename = args.filename
    verbosity = args.verbose
    start_frag = args.start_frag
    end_frag = args.end_frag
    no_anomaly = args.no_anomaly
    overwrite = args.overwrite

    linear = args.linear
    log = args.log

    # User didn't pass either flag, so default to both being true.
    if (not linear) and (not log):
        linear = True
        log = True

    data = trgtools.HSIReader(filename, verbosity)

    # Load all case.
    if start_frag == 0 and end_frag == -1:
        data.read_all_fragments()  # Has extra debug/warning info
    else:  # Only load some.
        if end_frag != 0:  # Python doesn't like [n:0]
            frag_paths = data.get_fragment_paths()[start_frag:end_frag]
        elif end_frag == 0:
            frag_paths = data.get_fragment_paths()[start_frag:]

        for path in frag_paths:
            data.read_fragment(path)

    # Find a new save name or overwrite an old one.
    save_name = find_save_name(data.run_id, data.file_index, overwrite)

    print(f"Number of HSIs: {len(data.hsi_data)}")  # Enforcing output for useful metric

    # Plotting

    if not no_anomaly:
        anomaly_filename = f"{save_name}.txt"
        print(f"Writing descriptive statistics to {anomaly_filename}.")
        if os.path.isfile(anomaly_filename):
            # Prepare a new ta_anomaly_summary.txt
            os.remove(anomaly_filename)

    # Dictionary containing unique title, xlabel, and xticks (only some)
    base_hist_dict = {
            'crate': {
                'title': "Crate Histogram",
                'xlabel': "Crate",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'detector_id': {
                'title': "Detector ID Histogram",
                'xlabel': "Detector ID",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'link': {
                'title': "Link Histogram",
                'xlabel': "Link",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'sequence': {
                'title': "Sequence Histogram",
                'xlabel': "Sequence",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'timestamp': {
                'title': "Timestamp Histogram",
                'xlabel': "Timestamp",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'trigger': {
                'title': "Trigger Histogram",
                'xlabel': "Trigger",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            },
            'version': {
                'title': "Version Histogram",
                'xlabel': "Version",
                'ylabel': "Count",
                'linear': linear,
                'linear_style': dict(color='#63ACBE', alpha=0.6, label='Linear'),
                'log': log,
                'log_style': dict(color='#EE442F', alpha=0.6, label='Log')
            }
    }

    # TODO: Implement link separation.
    unique_links = np.unique(data.hsi_data['link'])
    print("Unique links:", unique_links)
    pdf_plotter = PDFPlotter(f"{save_name}.pdf")
    pdf = pdf_plotter.get_pdf()

    for unique_link in unique_links:
        hsi_data = data.hsi_data[np.where(data.hsi_data['link'] == unique_link)]
        print("Length of hsi_data:", len(hsi_data))
        link_hist_dict = deepcopy(base_hist_dict)
        # Generic Plots
        for hsi_key in hsi_data.dtype.names:
            if hsi_key == 'input_low' or hsi_key == 'input_high':  # These have a special plot.
                continue
            link_hist_dict[hsi_key]['title'] = f"Link ID {unique_link} " + base_hist_dict[hsi_key]['title']
            if not no_anomaly:
                write_summary_stats(hsi_data[hsi_key], anomaly_filename, hsi_key + f" link {unique_link}")
            try:
                pdf_plotter.plot_histogram(hsi_data[hsi_key], link_hist_dict[hsi_key])
            except IndexError:
                print(f"WARNING: {hsi_key} on link {unique_link} tried an illegal index.")

        # Analysis Plots

        low_bitmap_dict = {
                'title': f"Link ID {unique_link} Input Low Bitmap",
                'xlabel': "Bit Position",
                'ylabel': "HSI Fragment",
                'imshow_style': {
                    'cmap': 'Greys_r',
                    'aspect': 'auto'
                },
                'colorbar': True,
                'grid': True,
                'entries': True,
                'entries_color': ('k', 'w')
        }
        high_bitmap_dict = {
                'title': f"Link ID {unique_link} Input High Bitmap",
                'xlabel': "Bit Position",
                'ylabel': "HSI Fragment",
                'imshow_style': {
                    'cmap': 'Greys_r',
                    'aspect': 'auto'
                },
                'colorbar': True,
                'grid': True,
                'entries': True
        }
        plot_pdf_bitmap(data.hsi_data['input_low'], pdf, low_bitmap_dict)
        plot_pdf_bitmap(data.hsi_data['input_high'], pdf, high_bitmap_dict)

    return None


if __name__ == "__main__":
    main()
