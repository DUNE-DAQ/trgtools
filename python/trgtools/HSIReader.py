"""
Reader class for HSI data.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import detdataformats

import numpy as np


class HSIReader(HDF5Reader):
    """
    """
    # HSI data type
    hsi_dt = np.dtype([
                        ('crate', np.uint32),
                        ('detector_id', np.uint32),
                        ('input_low', tuple),   # Original is uint32
                        ('input_high', tuple),  # Saving as bit positions
                        ('link', np.uint32),
                        ('sequence', np.uint32),
                        ('timestamp', np.uint64),
                        ('trigger', np.uint32),
                        ('version', np.uint32)
                      ])
    hsi_data = np.array([], dtype=hsi_dt)

    def __init__(self, filename: str, verbosity: int = 0) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            verbosity (int): Verbose level. 0: Only errors. 1: Warnings. 2: All.

        Returns nothing.
        """
        super().__init__(filename, verbosity)
        return None

    def _filter_fragment_paths(self) -> None:
        """ Fileter the fragment paths for HSIs. """
        fragment_paths = []

        # HSI fragment paths contain 'Hardware_Signal' in their path.
        for path in self._fragment_paths:
            if "Hardware_Signal" in path:
                fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def _get_bit_positions(self, bitmap: int) -> tuple:
        """ Get the bit positions for the given integer.

        Assumes that :bitmap: is a uint32.
        """
        if bitmap == 0:  # Edge case
            return tuple()

        sniff = 1
        bit_pos = []
        for pos in range(32):
            if bitmap & sniff:
                bit_pos.append(pos)
            sniff <<= 1
        return tuple(bit_pos)

    def read_fragment(self, fragment_path: str) -> np.ndarray:
        """
        Read from the given data fragment path.

        Returns a np.ndarray of the HSIs that were read and appends to
        :self.hsi_data:.
        """
        if self._verbosity >= 2:
            print("="*60)
            print(f"INFO: Reading from the path\n{fragment_path}")

        fragment = self._h5_file.get_frag(fragment_path)
        fragment_data_size = fragment.get_data_size()

        if fragment_data_size == 0:
            self._num_empty += 1
            if self._verbosity >= 1:
                print(
                        self._FAIL_TEXT_COLOR
                        + self._BOLD_TEXT
                        + "WARNING: Empty fragment. Returning empty array."
                        + self._END_TEXT_COLOR
                )
                print("="*60)
            return np.array([], dtype=self.hsi_dt)

        hsi_idx = 0  # Debugging output.
        byte_idx = 0  # Variable HSI sizing, must do while loop.
        hsi_datum = detdataformats.HSIFrame(fragment.get_data())
        np_hsi_datum = np.array([(
                             hsi_datum.crate,
                             hsi_datum.detector_id,
                             self._get_bit_positions(hsi_datum.input_low),
                             self._get_bit_positions(hsi_datum.input_high),
                             hsi_datum.link,
                             hsi_datum.sequence,
                             hsi_datum.get_timestamp(),
                             hsi_datum.trigger,
                             hsi_datum.version)],
                             dtype=self.hsi_dt)
        self.hsi_data = np.hstack((self.hsi_data, np_hsi_datum))

        if self._verbosity >= 2:
            print("INFO: Finished reading.")
            print("="*60)
        return np_hsi_datum

    def clear_data(self) -> None:
        self.hsi_data = np.array([], dtype=self.hsi_dt)
