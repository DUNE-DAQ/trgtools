"""
Reader class for TC data.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import trgdataformats

import numpy as np


class TCReader(HDF5Reader):
    """
    Class that reads a given HDF5 data file and can
    process the TC fragments within.

    Loading fragments appends to :self.tc_data: and :self.ta_data:.
    NumPy dtypes of :self.tc_data: and :self.ta_data: are available
    as :TCReader.tc_dt: and :TCReader.ta_dt:.

    TC reading will print any information that is relevant about the
    loading process. TO hide these prints, specify :quiet = True: on
    init.
    """
    # TC data type
    tc_dt = np.dtype([
        ('algorithm', np.uint8),
        ('detid', np.uint16),
        ('num_tas', np.uint64),  # Greedy
        ('time_candidate', np.uint64),
        ('time_end', np.uint64),
        ('time_start', np.uint64),
        ('type', np.uint8),
        ('version', np.uint16),
    ])
    tc_data = np.array([], dtype=tc_dt)  # Will concatenate new TCs

    # TA data type
    ta_dt = np.dtype([
        ('adc_integral', np.uint64),
        ('adc_peak', np.uint64),
        ('algorithm', np.uint8),
        ('channel_end', np.int32),
        ('channel_peak', np.int32),
        ('channel_start', np.int32),
        ('detid', np.uint16),
        ('time_activity', np.uint64),
        ('time_end', np.uint64),
        ('time_peak', np.uint64),
        ('time_start', np.uint64),
        ('type', np.uint8),
        ('version', np.uint16)
    ])
    ta_data = []  # ta_data[i] will be a np.ndarray of TAs from the i-th TC

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
        """ Filter the fragment paths for TCs. """
        fragment_paths = []

        # TC fragment paths contain their name in the path.
        for path in self._fragment_paths:
            if "Trigger_Candidate" in path:
                fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def read_fragment(self, fragment_path: str) -> np.ndarray:
        """
        Read from the given data fragment path.

        Returnss a np.ndarray of the TCs that were read and appends to :self.tc_data:.
        """
        if not self._quiet:
            print("="*60)
            print(f"INFO: Reading from the path\n{fragment_path}")

        fragment = self._h5_file.get_frag(fragment_path)
        fragment_data_size = fragment.get_data_size()

        if fragment_data_size == 0:  # Empty fragment
            self._num_empty += 1
            if not self._quiet:
                print(
                        self._FAIL_TEXT_COLOR
                        + self._BOLD_TEXT
                        + "WARNING: Empty fragment."
                        + self._END_TEXT_COLOR
                )
                print("="*60)
            return np.array([], dtype=self.tc_dt)

        tc_idx = 0  # Debugging output.
        byte_idx = 0  # Variable TC sizing, must do a while loop.
        while byte_idx < fragment_data_size:
            if not self._quiet:
                print(f"INFO: Fragment Index: {tc_idx}.")
                tc_idx += 1
                print(f"INFO: Byte Index / Frag Size: {byte_idx} / {fragment_data_size}")

            # Process TC data
            tc_datum = trgdataformats.TriggerCandidate(fragment.get_data_bytes(byte_idx))
            np_tc_datum = np.array([(
                                tc_datum.data.algorithm,
                                tc_datum.data.detid,
                                tc_datum.n_inputs(),
                                tc_datum.data.time_candidate,
                                tc_datum.data.time_end,
                                tc_datum.data.time_start,
                                tc_datum.data.type,
                                tc_datum.data.version)],
                                dtype=self.tc_dt)

            self.tc_data = np.hstack((self.tc_data, np_tc_datum))

            byte_idx += tc_datum.sizeof()
            if not self._quiet:
                print(f"Upcoming byte index: {byte_idx}.")

            # Process TA data
            np_ta_data = np.zeros(np_tc_datum['num_tas'], dtype=self.ta_dt)
            for ta_idx, ta in enumerate(tc_datum):
                np_ta_data[ta_idx] = np.array([(
                                            ta.adc_integral,
                                            ta.adc_peak,
                                            np.uint8(ta.algorithm),
                                            ta.channel_end,
                                            ta.channel_peak,
                                            ta.channel_start,
                                            np.uint16(ta.detid),
                                            ta.time_activity,
                                            ta.time_end,
                                            ta.time_peak,
                                            ta.time_start,
                                            np.uint8(ta.type),
                                            ta.version)],
                                            dtype=self.ta_dt)
            self.ta_data.append(np_ta_data)  # Jagged array

        if not self._quiet:
            print("INFO: Finished reading.")
            print("="*60)
        return np_tc_datum
