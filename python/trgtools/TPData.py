"""
Class containing TP data and the ability to process data.
"""

import os

import numpy as np

from hdf5libs import HDF5RawDataFile
import daqdataformats
import trgdataformats

class TPData:
    _FAIL_TEXT_COLOR = '\033[91m'
    _WARNING_TEXT_COLOR = '\033[93m'
    _BOLD_TEXT = '\033[1m'
    _END_TEXT_COLOR = '\033[0m'

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
        self._set_tp_frag_paths(self._h5_file.get_all_fragment_dataset_paths())
        self.tp_data = [] # Will have length == number of fragments
        self._quiet = quiet

    def _set_tp_frag_paths(self, frag_paths) -> None:
        """
        Only collect the fragment paths that are for TAs.
        """
        self._frag_paths = []
        for frag_path in frag_paths:
            if 'Trigger_Primitive' in frag_path:
                self._frag_paths.append(frag_path)

    def get_tp_frag_paths(self) -> list:
        return self._frag_paths

    def load_frag(self, frag_path) -> np.ndarray:
        tp = self._h5_file.get_frag(frag_path)

        data_size = tp.get_data_size()
        tp_size = trgdataformats.TriggerPrimitive.sizeof()
        num_tps = data_size // tp_size
        print("Load frag, num_tps:", num_tps)

        # TODO: Change name to frag_data since this object is all TPs in a frag
        np_tp_datum = np.zeros((num_tps,), dtype=self.tp_dt)
        for idx, tp_idx in enumerate(range(0, data_size, tp_size)):
            tp_datum = trgdataformats.TriggerPrimitive(tp.get_data(tp_idx))

            np_tp_datum[idx] = np.array([(
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
        self.tp_data.append(np_tp_datum)

        return np_tp_datum

    def load_all_frags(self) -> None:
        """
        Load all of the fragments to the current object.
        """
        for idx, frag_path in enumerate(self._frag_paths):
            _ = self.load_frag(frag_path) # Only care about the append to self.tp_data
