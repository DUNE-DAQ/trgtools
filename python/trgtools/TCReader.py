"""
Reader class for TC data.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import trgdataformats

import numpy as np
from numpy.typing import NDArray


class TCReader(HDF5Reader):
    """
    Class that reads a given HDF5 data file and can
    process the TC fragments within.

    Loading fragments appends to :self.tc_data: and :self.ta_data:.
    NumPy dtypes of :self.tc_data: and :self.ta_data: are available
    as :TCReader.tc_dt: and :TCReader.ta_dt:.

    TC reading can print information that is relevant about the
    loading process by specifying the verbose level. 0 for errors
    only. 1 for warnings. 2 for all information.
    """
    # TC data type
    tc_dt = np.dtype([
        ('algorithm', trgdataformats.TriggerCandidateData.Algorithm),
        ('detid', np.uint16),
        ('num_tas', np.uint64),  # Greedy
        ('time_candidate', np.uint64),
        ('time_end', np.uint64),
        ('time_start', np.uint64),
        ('type', trgdataformats.TriggerCandidateData.Type),
        ('version', np.uint16),
        ('trigger_number', np.uint64)
    ])

    # TA data type
    ta_dt = np.dtype([
        ('adc_integral', np.uint64),
        ('adc_peak', np.uint64),
        ('algorithm', trgdataformats.TriggerActivityData.Algorithm),
        ('channel_end', np.int32),
        ('channel_peak', np.int32),
        ('channel_start', np.int32),
        ('detid', np.uint16),
        ('time_activity', np.uint64),
        ('time_end', np.uint64),
        ('time_peak', np.uint64),
        ('time_start', np.uint64),
        ('type', trgdataformats.TriggerActivityData.Type),
        ('version', np.uint16)
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
        self.tc_data = np.array([], dtype=self.tc_dt)  # Will concatenate new TCs
        self.ta_data = []  # ta_data[i] will be a np.ndarray of TAs from the i-th TC
        return None

    def __getitem__(self, key: int | str) -> NDArray[tc_dt]:
        return self.tc_data[key]

    def __setitem__(self, key: int | str, value: NDArray[tc_dt]) -> None:
        self.tc_data[key] = value
        return

    def __len__(self) -> int:
        return len(self.tc_data)

    def _filter_fragment_paths(self) -> None:
        """ Filter the fragment paths for TCs. """
        fragment_paths = []

        # TC fragment paths contain their name in the path.
        for path in self._fragment_paths:
            if "Trigger_Candidate" in path:
                fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def read_fragment(self, fragment_path: str) -> NDArray:
        """
        Read from the given data fragment path.

        Returns a np.ndarray of the first TC that was read and appends all TCs in the fragment to :self.tc_data:.
        """
        if self._verbosity >= 2:
            print("="*60)
            print(f"INFO: Reading from the path\n{fragment_path}")

        fragment = self._h5_file.get_frag(fragment_path)
        fragment_data_size = fragment.get_data_size()
        trigger_number = fragment.get_trigger_number()

        if fragment_data_size == 0:  # Empty fragment
            self._num_empty += 1
            if self._verbosity >= 1:
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
            if self._verbosity >= 2:
                print(f"INFO: Fragment Index: {tc_idx}.")
                tc_idx += 1
                print(f"INFO: Byte Index / Frag Size: {byte_idx} / {fragment_data_size}")

            # Process TC data
            tc_datum = trgdataformats.TriggerCandidate(fragment.get_data(byte_idx))
            np_tc_datum = np.array([(
                                tc_datum.data.algorithm,
                                tc_datum.data.detid,
                                tc_datum.n_inputs(),
                                tc_datum.data.time_candidate,
                                tc_datum.data.time_end,
                                tc_datum.data.time_start,
                                tc_datum.data.type,
                                tc_datum.data.version,
                                trigger_number)],
                                dtype=self.tc_dt)

            self.tc_data = np.hstack((self.tc_data, np_tc_datum))

            byte_idx += tc_datum.sizeof()
            if self._verbosity >= 2:
                print(f"INFO: Upcoming byte index: {byte_idx}.")

            # Process TA data
            np_ta_data = np.zeros(np_tc_datum['num_tas'], dtype=self.ta_dt)
            for ta_idx, ta in enumerate(tc_datum):
                np_ta_data[ta_idx] = np.array([(
                                            ta.adc_integral,
                                            ta.adc_peak,
                                            ta.algorithm,
                                            ta.channel_end,
                                            ta.channel_peak,
                                            ta.channel_start,
                                            np.uint16(ta.detid),
                                            ta.time_activity,
                                            ta.time_end,
                                            ta.time_peak,
                                            ta.time_start,
                                            ta.type,
                                            ta.version)],
                                            dtype=self.ta_dt)
            self.ta_data.append(np_ta_data)  # Jagged array

        if self._verbosity >= 2:
            print("INFO: Finished reading.")
            print("="*60)
        return np_tc_datum

    def clear_data(self) -> None:
        self.tc_data = np.array([], dtype=self.tc_dt)
        self.ta_data = []
