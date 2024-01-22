"""
Class containing TA data and the ability to process data.
"""
import os

import numpy as np

from hdf5libs import HDF5RawDataFile
import daqdataformats
import trgdataformats

class TAData:
    _FAIL_TEXT_COLOR = '\033[91m'
    _WARNING_TEXT_COLOR = '\033[93m'
    _BOLD_TEXT = '\033[1m'
    _END_TEXT_COLOR = '\033[0m'

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
        self._h5_file = HDF5RawDataFile(filename)
        self._set_ta_frag_paths(self._h5_file.get_all_fragment_dataset_paths())
        self.ta_data = np.zeros((len(self._frag_paths),), dtype=self.ta_dt)
        self.tp_data = []
        self._quiet = quiet
        self._nonempty_frags_mask = np.ones((len(self._frag_paths),), dtype=bool)
        if "run" in filename:
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

    def load_frag(self, frag_path, index=None) -> (np.ndarray, np.ndarray):
        ta = self._h5_file.get_frag(frag_path)
        if ta.get_data_size() == 0:
            if not self._quiet:
                print(self._FAIL_TEXT_COLOR + self._BOLD_TEXT + "WARNING: Empty fragment. Returning empty array." + self._END_TEXT_COLOR)
                print(self._WARNING_TEXT_COLOR + self._BOLD_TEXT + f"INFO: Fragment Path: {frag_path}" + self._END_TEXT_COLOR)
            if index != None:
                self._nonempty_frags_mask[index] = False
            return np.zeros((0,), dtype=self.ta_dt), np.zeros((0,), dtype=self.tp_dt)
        if index == None:
            index = self._frag_paths.index(frag_path)

        ta_datum = trgdataformats.TriggerActivity(ta.get_data())
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
                                dtype=self.ta_dt)

        num_tps = ta_datum.n_inputs()
        np_tp_datum = np.zeros((num_tps,), dtype=self.tp_dt)
        for idx, tp in enumerate(ta_datum):
            np_tp_datum[idx] = np.array([(
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
        self.tp_data.append(np_tp_datum)
        self.ta_data[index] = np_ta_datum

        return np_ta_datum, np_tp_datum

    def _filter_frags(self) -> None:
        self.ta_data = self.ta_data[self._nonempty_frags_mask]

    def load_all_frags(self) -> None:
        miscount = 0
        for idx, frag_path in enumerate(self._frag_paths):
            ta_datum, _ = self.load_frag(frag_path, idx)
            if len(ta_datum) == 0:
                miscount += 1
        if miscount != 0 and not self._quiet:
            print(self._FAIL_TEXT_COLOR + self._BOLD_TEXT + f"WARNING: Skipped {miscount} frags." + self._END_TEXT_COLOR)
            print(self._WARNING_TEXT_COLOR + "INFO: Filtering skipped fragments." + self._END_TEXT_COLOR)
            self._filter_frags()
