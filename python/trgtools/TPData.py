"""
Class containing TP data and the ability to process data.
"""

import os

import numpy as np

from hdf5libs import HDF5RawDataFile
import daqdataformats
import trgdataformats

class TPData:
    """
    Class that loads a given TPStream file and can
    process the TP fragments within.

    Loading fragments populates obj.tp_data. The
    Numpy dtype is available on obj.tp_dt.
    """
    ## Useful print colors
    _FAIL_TEXT_COLOR = '\033[91m'
    _WARNING_TEXT_COLOR = '\033[93m'
    _BOLD_TEXT = '\033[1m'
    _END_TEXT_COLOR = '\033[0m'

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
        Loads the given HDF5 file and inits member data.
        """
        self._h5_file = HDF5RawDataFile(filename)
        self._set_tp_frag_paths(self._h5_file.get_all_fragment_dataset_paths())
        self.tp_data = [] # Will have length == number of fragments
        self._quiet = quiet

        # File identification attributes
        self.run_id = self._h5_file.get_int_attribute('run_number')
        self.file_index = self._h5_file.get_int_attribute('file_index')

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
        """
        Load a fragment from a given fragment path
        and append to self.tp_data.

        Returns an np.ndarray of dtype self.tp_dt.
        """
        frag = self._h5_file.get_frag(frag_path)

        data_size = frag.get_data_size()
        tp_size = trgdataformats.TriggerPrimitive.sizeof()
        num_tps = data_size // tp_size
        if (not self._quiet):
            print(f"Loaded frag with {num_tps} TPs.")

        np_tp_data = np.zeros((num_tps,), dtype=self.tp_dt)
        for idx, byte_idx in enumerate(range(0, data_size, tp_size)): # Increment by TP size
            tp_datum = trgdataformats.TriggerPrimitive(frag.get_data(byte_idx))

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
        self.tp_data.append(np_tp_data)

        return np_tp_data

    def load_all_frags(self) -> None:
        """
        Load all of the fragments to the current object.
        """
        for idx, frag_path in enumerate(self._frag_paths):
            _ = self.load_frag(frag_path) # Only care about the append to self.tp_data
