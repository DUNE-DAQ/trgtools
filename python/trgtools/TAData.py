"""
Class containing TA data and the ability to process data.
"""
import os

import numpy as np

from hdf5libs import HDF5RawDataFile
import daqdataformats
import trgdataformats

class TAData:
    """
    Class that loads a given TPStream file and can
    process the TA fragments within.

    Loading fragments populates obj.ta_data and obj.tp_data.
    Numpy dtypes of ta_data and tp_data are available as
    obj.ta_dt and obj.tp_dt.

    Gives warnings and information when trying to load
    empty fragments. Can be quieted with quiet=True on
    init.
    """
    ## Useful print colors
    _FAIL_TEXT_COLOR = '\033[91m'
    _WARNING_TEXT_COLOR = '\033[93m'
    _BOLD_TEXT = '\033[1m'
    _END_TEXT_COLOR = '\033[0m'

    ## TA data type
    ta_dt = np.dtype([
                      ('adc_integral', np.uint64),
                      ('adc_peak', np.uint64),
                      ('algorithm', np.uint8),
                      ('channel_end', np.int32),
                      ('channel_peak', np.int32),
                      ('channel_start', np.int32),
                      ('detid', np.uint16),
                      ('num_tps', np.uint64), # Greedy
                      ('time_activity', np.uint64),
                      ('time_end', np.uint64),
                      ('time_peak', np.uint64),
                      ('time_start', np.uint64),
                      ('type', np.uint8)
                     ])
    ## TP data type
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

    def __init__(self, filename, quiet=False):
        """
        Loads the given HDF5 file and inits memmber data.
        """
        self._h5_file = HDF5RawDataFile(filename)
        self._set_ta_frag_paths(self._h5_file.get_all_fragment_dataset_paths())
        self._quiet = quiet

        self.ta_data = np.array([], dtype=self.ta_dt).reshape(0,1) # Will concatenate new TAs
        self.tp_data = [] # tp_data[i] will be a np.ndarray of TPs from the i-th TA
        self._ta_size = trgdataformats.TriggerActivityOverlay().sizeof()

        # Masking frags that are found as empty.
        self._nonempty_frags_mask = np.ones((len(self._frag_paths),), dtype=bool)
        self._num_empty = 0

        if "run" in filename: # Waiting on hdf5libs PR to use get_int_attribute
            tmp_name = filename.split("run")[1]
            try:
                self.run_id = int(tmp_name.split("_")[0])
            except:
                if not self._quiet:
                    print(self._WARNING_TEXT_COLOR + "WARNING: Couldn't find Run ID in file name. Using run id 0." + self._END_TEXT_COLOR)
                self.run_id = 0
            try:
                self.sub_run_id = int(tmp_name.split("_")[1])
            except:
                if not self._quiet:
                    print(self._WARNING_TEXT_COLOR + "WARNING: Couldn't find SubRun ID in file name. Using SubRun ID 1000." + self._END_TEXT_COLOR)
                self.sub_run_id = 1000
        else:
            if not self._quiet:
                print(self._WARNING_TEXT_COLOR + "WARNING: Couldn't find Run ID in file name. Using run id 0." + self._END_TEXT_COLOR)
                print(self._WARNING_TEXT_COLOR + "WARNING: Couldn't find SubRun ID in file name. Using SubRun ID 1000." + self._END_TEXT_COLOR)
            self.run_id = 0
            self.sub_run_id = 1000

    def _set_ta_frag_paths(self, frag_paths) -> None:
        """
        Only collect the fragment paths that are for TAs.
        """
        self._frag_paths = []
        for frag_path in frag_paths:
            if 'Trigger_Activity' in frag_path:
                self._frag_paths.append(frag_path)

    def get_ta_frag_paths(self) -> list:
        return self._frag_paths

    def load_frag(self, frag_path, index=None) -> None:
        """
        Load a fragment from a given fragment path.
        Saves the results to self.ta_data and self.tp_data.
        Returns nothing.
        """
        frag = self._h5_file.get_frag(frag_path)
        frag_data_size = frag.get_data_size()
        if frag_data_size == 0:
            self._num_empty += 1
            if not self._quiet:
                print(self._FAIL_TEXT_COLOR + self._BOLD_TEXT + "WARNING: Empty fragment. Returning empty array." + self._END_TEXT_COLOR)
                print(self._WARNING_TEXT_COLOR + self._BOLD_TEXT + f"INFO: Fragment Path: {frag_path}" + self._END_TEXT_COLOR)
            if index != None:
                self._nonempty_frags_mask[index] = False
            return
        if index == None:
            index = self._frag_paths.index(frag_path)

        ta_idx = 0 # Only used to output.
        byte_idx = 0 # Variable TA sizing, must do while loop.
        while (byte_idx < frag_data_size):
            if (not self._quiet):
                print(f"Fragment Index: {ta_idx}.")
                ta_idx += 1
                print(f"Byte Index / Frag Size: {byte_idx} / {frag_data_size}")
            ## Process TA data
            ta_datum = trgdataformats.TriggerActivity(frag.get_data(byte_idx))
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
                                    np.uint8(ta_datum.data.type))],
                                    dtype=self.ta_dt).reshape(1,1)
            self.ta_data = np.vstack((self.ta_data, np_ta_datum))
            byte_idx += ta_datum.sizeof()
            if (not self._quiet):
                print(f"Upcoming byte: {byte_idx}")

            ## Process TP data
            np_tp_data = np.zeros(np_ta_datum['num_tps'][0], dtype=self.tp_dt)
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
            self.tp_data.append(np_tp_data) # Jagged array

        return

    def load_all_frags(self) -> None:
        """
        Load all fragments.

        Returns nothing. Data is accessible from obj.ta_data
        and obj.tp_data.
        """
        for idx, frag_path in enumerate(self._frag_paths):
            self.load_frag(frag_path, idx)
        if self._num_empty != 0 and not self._quiet:
            print(self._FAIL_TEXT_COLOR + self._BOLD_TEXT + f"WARNING: Skipped {self._num_empty} frags." + self._END_TEXT_COLOR)
