"""
Reader class for TP data.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import trgdataformats

import numpy as np


class TPReader(HDF5Reader):
    """
    Class that reads a given HDF5 data file and can
    process the TP fragments within.

    Loading fragments appends to :self.tp_data:. The
    NumPy dtypes of :self.tp_data: is available as
    :TPReader.tp_dt:.

    TP reading can print information that is relevant about the
    loading process by specifying the verbose level. 0 for errors
    only. 1 for warnings. 2 for all information.
    """
    # TP data type
    tp_dt = np.dtype([
                      ('adc_integral', np.uint32),
                      ('adc_peak', np.uint32),
                      ('algorithm', np.uint8),
                      ('channel', np.int32),
                      ('detid', np.uint16),
                      ('flag', np.uint16),
                      ('time_over_threshold', np.uint64),
                      ('time_peak', np.uint64),
                      ('time_start', np.uint64),
                      ('type', np.uint8),
                      ('version', np.uint16)
                     ])

    def __init__(self, filename: str, verbosity: int = 0) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            verbosity (int): Verbose level. 0: Only errors. 1: Warnings. 2: All.

        Returns nothing.
        """
        super().__init__(filename, verbosity)
        self.tp_data = np.array([], dtype=self.tp_dt)
        return None

    def _filter_fragment_paths(self) -> None:
        """ Filter the fragment paths for TAs. """
        fragment_paths = []

        # TA fragment paths contain their name in the path.
        for path in self._fragment_paths:
            if "Trigger_Primitive" in path:
                fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def read_fragment(self, fragment_path: str) -> np.ndarray:
        """
        Read from the given data fragment path.

        Returns a np.ndarray of the TPs that were read and appends to
        :self.tp_data:.
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
            return np.array([], dtype=self.tp_dt)

        tp_size = trgdataformats.TriggerPrimitive.sizeof()
        num_tps = fragment_data_size // tp_size
        if self._verbosity >= 2:
            print(f"INFO: Loaded fragment with {num_tps} TPs.")

        np_tp_data = np.zeros((num_tps,), dtype=self.tp_dt)
        for idx, byte_idx in enumerate(range(0, fragment_data_size, tp_size)):  # Increment by TP size
            tp_datum = trgdataformats.TriggerPrimitive(fragment.get_data(byte_idx))

            np_tp_data[idx] = np.array([(
                                    tp_datum.adc_integral,
                                    tp_datum.adc_peak,
                                    tp_datum.algorithm,
                                    tp_datum.channel,
                                    tp_datum.detid,
                                    tp_datum.flag,
                                    tp_datum.time_over_threshold,
                                    tp_datum.time_peak,
                                    tp_datum.time_start,
                                    tp_datum.type,
                                    tp_datum.version)],
                                    dtype=self.tp_dt)
        self.tp_data = np.hstack((self.tp_data, np_tp_data))

        if self._verbosity >= 2:
            print("INFO: Finished reading.")
            print("="*60)
        return np_tp_data

    def clear_data(self) -> None:
        self.tp_data = np.array([], dtype=self.tp_dt)
