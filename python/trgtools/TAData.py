"""
Class containing TA data and the ability to process data.
"""
import os

import numpy as np

from hdf5libs import HDF5RawDataFile
import daqdataformats
import trgdataformats

class TAData:
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

    def __init__(self, filename):
        self._h5_file = HDF5RawDataFile(filename)
        self._set_ta_frag_paths(self._h5_file.get_all_fragment_dataset_paths())
        self.ta_data = np.zeros((len(self._frag_paths),), dtype=self.ta_dt)
        self.run_id = int(filename.split("_")[1][3:])
        self.sub_run_id = int(filename.split("_")[2][:4])

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

    def _set_entry(self, data_row, ta_datum) -> None:
            data_row = np.array([(
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

    def load_all_frags(self) -> None:
        for idx, frag_path in enumerate(self._frag_paths):
            ta = self._h5_file.get_frag(frag_path)
            ta_datum = trgdataformats.TriggerActivity(ta.get_data())
            self._set_entry(self.ta_data[idx], ta_datum)
