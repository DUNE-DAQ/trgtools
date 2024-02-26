"""
Reader class for TA data.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import trgdataformats

import numpy as np


class TAReader(HDF5Reader):
    """
    Class that reads a given HDF5 data file and can
    process the TA fragments within.

    Loading fragments appends to :self.ta_data: and :self.tp_data:.
    NumPy dtypes of :self.ta_data: and :self.tp_data: are available
    as :TAReader.ta_dt: and :TAReader.tp_dt:.

    TA reading will print any information that is relevant about the
    loading process. To hide these prints, specify :quiet = True: on
    init.
    """
    # TA data type
    ta_dt = np.dtype([
                      ('adc_integral', np.uint64),
                      ('adc_peak', np.uint64),
                      ('algorithm', np.uint8),
                      ('channel_end', np.int32),
                      ('channel_peak', np.int32),
                      ('channel_start', np.int32),
                      ('detid', np.uint16),
                      ('num_tps', np.uint64),  # Greedy
                      ('time_activity', np.uint64),
                      ('time_end', np.uint64),
                      ('time_peak', np.uint64),
                      ('time_start', np.uint64),
                      ('type', np.uint8),
                      ('version', np.uint16)
                     ])
    ta_data = np.array([], dtype=ta_dt)

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
    tp_data = []

    def __init__(self, filename: str, quiet: bool = False) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            quiet (bool): Quiets outputs if true.

        Returns nothing.
        """
        super().__init__(filename, quiet)
        return None

    def _filter_fragment_paths(self) -> None:
        """ Filter the fragment paths for TAs. """
        fragment_paths = []

        # TA fragment paths contain their name in the path.
        for path in self._fragment_paths:
            if "Trigger_Activity" in path:
                fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def read_fragment(self, fragment_path: str) -> np.ndarray:
        """
        Read from the given data fragment path.

        Returns a np.ndarray of the TAs that were read and appends to
        :self.ta_data:.
        """
        if not self._quiet:
            print("="*60)
            print(f"INFO: Reading from the path\n{fragment_path}")

        fragment = self._h5_file.get_frag(fragment_path)
        fragment_data_size = fragment.get_data_size()

        if fragment_data_size == 0:
            self._num_empty += 1
            if not self._quiet:
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
            if not self._quiet:
                print(f"INFO: Fragment Index: {ta_idx}.")
                ta_idx += 1
                print(f"INFO: Byte Index / Frag Size: {byte_idx} / {fragment_data_size}")

            # Read TA data
            ta_datum = trgdataformats.TriggerActivity(fragment.get_data(byte_idx))
            np_ta_datum = np.array([(
                                ta_datum.data.adc_integral,
                                ta_datum.data.adc_peak,
                                np.uint8(ta_datum.data.algorithm),
                                ta_datum.data.channel_end,
                                ta_datum.data.channel_peak,
                                ta_datum.data.channel_start,
                                np.uint16(ta_datum.data.detid),
                                ta_datum.n_inputs(),
                                ta_datum.data.time_activity,
                                ta_datum.data.time_end,
                                ta_datum.data.time_peak,
                                ta_datum.data.time_start,
                                np.uint8(ta_datum.data.type),
                                np.uint16(ta_datum.data.version))],
                                dtype=self.ta_dt)

            self.ta_data = np.hstack((self.ta_data, np_ta_datum))

            byte_idx += ta_datum.sizeof()
            if not self._quiet:
                print(f"Upcoming byte index: {byte_idx}")

            # Process TP data
            np_tp_data = np.zeros(np_ta_datum['num_tps'], dtype=self.tp_dt)
            for tp_idx, tp in enumerate(ta_datum):
                np_tp_data[tp_idx] = np.array([(
                                            tp.adc_integral,
                                            tp.adc_peak,
                                            tp.algorithm,
                                            tp.channel,
                                            tp.detid,
                                            tp.flag,
                                            tp.time_over_threshold,
                                            tp.time_peak,
                                            tp.time_start,
                                            tp.type,
                                            tp.version)],
                                            dtype=self.tp_dt)
            self.tp_data.append(np_tp_data)  # Jagged array

        if not self._quiet:
            print("INFO: Finished reading.")
            print("="*60)
        return np_ta_datum
