"""
TriggerCandidate reader class to read and store data
from an HDF5 file.
"""
import daqdataformats  # Not directly used, but necessary to interpret formats.
from hdf5libs import HDF5RawDataFile
import trgdataformats

import numpy as np

import os


class TCData():
    """
    Class that reads a given HDF5 TPStream file and stores
    the TC fragments within

    Loading fragments populates obj.tc_data and obj.ta_data.
    Numpy dtypes of ta_data and tp_data are available as
    obj.ta_dt and obj.tp_dt.

    Gives warnings and information when trying to load
    empty fragments. Can be quieted with quiet=True on
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

    tc_data = np.array([], dtype=tc_dt)  # Will concatenate new TCs
    ta_data = []  # ta_data[i] will be a np.ndarray of TAs from the i-th TC
    _tc_size = trgdataformats.TriggerCandidateOverlay().sizeof()
    _num_empty = 0

    def __init__(self, filename: str, quiet: bool = False) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            quiet (bool): Quiets outputs if true.
        """
        self._h5_file = HDF5RawDataFile(os.path.expanduser(filename))
        self._set_tc_frag_paths(self._h5_file.get_all_fragment_dataset_paths())
        self._quiet = quiet
        self.run_id = self._h5_file.get_int_attribute('run_number')
        self.file_index = self._h5_file.get_int_attribute('file_index')

        return None

    # TODO STYLE: Conforming to TAData and TPData, but this should be 'filter' not 'set'.
    def _set_tc_frag_paths(self, fragment_paths: list[str]) -> None:
        """
        Filter fragment paths for TriggerCandidates.

        Parameters:
            fragment_paths (list[str]): List of fragment paths that can be loaded from.
        Returns:
            Nothing. Creates a new instance member self._fragment_paths list[str].
        """
        self._fragment_paths = []
        for fragment_path in fragment_paths:
            if 'Trigger_Candidate' in fragment_path:
                self._fragment_paths.append(fragment_path)

        return None

    def get_tc_frag_paths(self) -> list[str]:
        """ Returns the list of fragment paths. """
        return self._fragment_paths

    # TODO STYLE: Conforming to TAData and TPData, but this should be 'read_fragment'.
    def load_frag(self, fragment_path: str) -> None:
        """
        Read from the given fragment path and store.

        Parameters:
            fragment_path (str): Fragment path to load from the initialized HDF5.
        Returns:
            Nothing. Stores the result in instance members :tc_data: and :ta_data:.
        """
        frag = self._h5_file.get_frag(fragment_path)
        frag_data_size = frag.get_data_size()

        if frag_data_size == 0:  # Empty fragment
            self._num_empty += 1
            if not self._quiet:
                print(
                        self._FAIL_TEXT_COLOR
                        + self._BOLD_TEXT
                        + "WARNING: Empty fragment."
                        + self._END_TEXT_COLOR
                )
                print(
                        self._WARNING_TEXT_COLOR
                        + self._BOLD_TEXT
                        + f"INFO: Fragment Path: {fragment_path}"
                        + self._END_TEXT_COLOR
                )

        tc_idx = 0  # Debugging output.
        byte_idx = 0  # Variable TC sizing, must do a while loop.
        while byte_idx < frag_data_size:
            if not self._quiet:
                print(f"INFO: Fragment Index: {tc_idx}.")
                tc_idx += 1
                print(f"INFO: Byte Index / Frag Size: {byte_idx} / {frag_data_size}")

            # Process TC data
            tc_datum = trgdataformats.TriggerCandidate(frag.get_data_bytes(byte_idx))
            np_tc_datum = np.array([(
                tc_datum.data.algorithm,
                tc_datum.data.detid,
                tc_datum.n_inputs(),
                tc_datum.data.time_candidate,
                tc_datum.data.time_end,
                tc_datum.data.time_start,
                tc_datum.data.type,
                tc_datum.data.version)],
                dtype=self.tc_dt
            )
            self.tc_data = np.hstack((self.tc_data, np_tc_datum))

            byte_idx += tc_datum.sizeof()
            if not self._quiet:
                print(f"Upcoming byte index: {byte_idx}.")

            # Process TA data
            np_ta_data = np.zeros(np_tc_datum['num_tas'], dtype=self.ta_dt)
            for ta_idx, ta in enumerate(tc_datum):
                np_ta_data[ta_idx] = np.array(
                    [(
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
                        ta.version
                    )],
                    dtype=self.ta_dt
                )
            self.ta_data.append(np_ta_data)  # Jagged array

        return None

    # TODO STYLE: Conforming to TAData and TPData, but this should be 'read_all_fragments'.
    def load_all_frags(self) -> None:
        """
        Read from all fragments in :self._fragment_paths:.

        Returns:
            Nothing. Stores the result in instance members :tc_data: and :ta_data:.
        """
        for frag_path in self._fragment_paths:
            self.load_frag(frag_path)
        if self._num_empty != 0 and not self._quiet:
            print(
                self._FAIL_TEXT_COLOR
                + self._BOLD_TEXT
                + f"WARNING: Skipped {self._num_empty} frags."
                + self._END_TEXT_COLOR
            )
