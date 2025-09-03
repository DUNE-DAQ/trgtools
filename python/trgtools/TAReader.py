"""
Reader class for TA data.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import trgdataformats

import numpy as np
from numpy.typing import NDArray


class TAReader(HDF5Reader):
    """
    Class that reads a given HDF5 data file and can
    process the TA fragments within.

    Loading fragments appends to :self.ta_data: and :self.tp_data:.
    NumPy dtypes of :self.ta_data: and :self.tp_data: are available
    as :TAReader.ta_dt: and :TAReader.tp_dt:.

    TA reading can print information that is relevant about the
    loading process by specifying the verbose level. 0 for errors
    only. 1 for warnings. 2 for all information.
    """
    # TA data type
    ta_dt = np.dtype([
                      ('adc_integral', np.uint64),
                      ('adc_peak', np.uint64),
                      ('algorithm', trgdataformats.TriggerActivityData.Algorithm),
                      ('channel_end', np.int32),
                      ('channel_peak', np.int32),
                      ('channel_start', np.int32),
                      ('detid', np.uint16),
                      ('num_tps', np.uint64),  # Greedy
                      ('time_activity', np.uint64),
                      ('time_end', np.uint64),
                      ('time_peak', np.uint64),
                      ('time_start', np.uint64),
                      ('type', trgdataformats.TriggerActivityData.Type),
                      ('version', np.uint16),
                      ('trigger_number', np.uint64)
                     ])

    # TP data type
    tp_dt = np.dtype([
                      ('adc_integral', np.uint32),
                      ('adc_peak', np.uint16),
                      ('channel', np.uint32),
                      ('detid', np.uint8),
                      ('flag', np.uint8),
                      ('samples_over_threshold', np.uint16),
                      ('samples_to_peak', np.uint16),
                      ('time_start', np.uint64),
                      ('version', np.uint8)
                     ])

    def __init__(self, filename: str, verbosity: int = 0, batch_mode: bool = False) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            verbosity (int): Verbose level. 0: Only errors. 1: Warnings. 2: All.

        Returns nothing.
        """
        super().__init__(filename, verbosity, batch_mode)
        self.ta_data = np.array([], dtype=self.ta_dt)
        self.tp_data = []
        return None

    def __getitem__(self, key: int | str) -> NDArray[ta_dt]:
        return self.ta_data[key]

    def __setitem__(self, key: int | str, value: NDArray[ta_dt]) -> None:
        self.ta_data[key] = value
        return

    def __len__(self) -> int:
        return len(self.ta_data)

    def _filter_fragment_paths(self) -> None:
        """ Filter the fragment paths for TAs. """
        fragment_paths = []

        # TA fragment paths contain their name in the path.
        for path in self._fragment_paths:
            if "Trigger_Activity" in path:
                fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def read_fragment(self, fragment_path: str) -> NDArray:
        """
        Read from the given data fragment path.

        Returns a np.ndarray of the first TA that was read and appends all TAs in the fragment to :self.ta_data:.
        """
        if self._verbosity >= 2:
            print("="*60)
            print(f"INFO: Reading from the path\n{fragment_path}")

        fragment = self._h5_file.get_frag(fragment_path)
        fragment_data_size = fragment.get_data_size()
        trigger_number = fragment.get_trigger_number()

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
            return np.array([], dtype=self.ta_dt)

        ta_idx = 0  # Debugging output.
        byte_idx = 0  # Variable TA sizing, must do while loop.
        while byte_idx < fragment_data_size:
            if self._verbosity >= 2:
                print(f"INFO: Fragment Index: {ta_idx}.")
                ta_idx += 1
                print(f"INFO: Byte Index / Frag Size: {byte_idx} / {fragment_data_size}")

            # Read TA data
            ta_datum = trgdataformats.TriggerActivity(fragment.get_data(byte_idx))
            np_ta_datum = np.array([(
                                ta_datum.data.adc_integral,
                                ta_datum.data.adc_peak,
                                ta_datum.data.algorithm,
                                ta_datum.data.channel_end,
                                ta_datum.data.channel_peak,
                                ta_datum.data.channel_start,
                                np.uint16(ta_datum.data.detid),
                                ta_datum.n_inputs(),
                                ta_datum.data.time_activity,
                                ta_datum.data.time_end,
                                ta_datum.data.time_peak,
                                ta_datum.data.time_start,
                                ta_datum.data.type,
                                np.uint16(ta_datum.data.version),
                                trigger_number)],
                                dtype=self.ta_dt)

            self.ta_data = np.hstack((self.ta_data, np_ta_datum))

            byte_idx += ta_datum.sizeof()
            if self._verbosity >= 2:
                print(f"INFO: Upcoming byte index: {byte_idx}")

            # Process TP data
            np_tp_data = np.zeros(np_ta_datum['num_tps'], dtype=self.tp_dt)
            for tp_idx, tp in enumerate(ta_datum):
                np_tp_data[tp_idx] = np.array([(
                                            tp.adc_integral,
                                            tp.adc_peak,
                                            tp.channel,
                                            tp.detid,
                                            tp.flag,
                                            tp.samples_over_threshold,
                                            tp.samples_to_peak,
                                            tp.time_start,
                                            tp.version)],
                                            dtype=self.tp_dt)
            self.tp_data.append(np_tp_data)  # Jagged array

        if self._verbosity >= 2:
            print("INFO: Finished reading.")
            print("="*60)
        return np_ta_datum

    def clear_data(self) -> None:
        self.ta_data = np.array([], dtype=self.ta_dt)
        self.tp_data = []
