"""
Reader class for WIBEth fragments.
"""
from .HDF5Reader import HDF5Reader

import daqdataformats  # noqa: F401 : Not used, but needed to recognize formats.
import detchannelmaps
from rawdatautils.unpack.utils import WIBEthUnpacker

import numpy as np


class WIBEthReader(HDF5Reader):
    """
    Class that reads a given HDF5 data file and can
    process the WIBEth fragments within.

    Reading a fragment gets returned and stored as the current
    data members :adc_data: and :timestamp_data:. Reading more
    fragments overwrites this data member. Both are NumPy arrays
    with matrix and vector representations, respectively.

    WIBEth reading can print information that is relevant about
    the loading process by specifying the verbose level. 0 for
    errors only. 1 for warnings. 2 for all information
    """

    def __init__(self, filename: str, channel_map: str, verbosity: int = 0) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            channel_map (str): Name of the channel map to use.
            verbosity (int): Verbose level. 0: Only errors. 1: Warnings. 2: All.

        Returns nothing.
        """
        super().__init__(filename, verbosity)
        self.adc_data = np.array([])
        self.timestamp_data = np.array([])
        self.channel_data = np.array([])
        self.channel_map = channel_map
        return None

    def _filter_fragment_paths(self) -> None:
        """ Filter the fragment paths for WIBEth. """
        fragment_paths =  []

        # WIBEth paths contain their name in the path.
        for path in self._fragment_paths:
            if "WIBEth" in path:
                    fragment_paths.append(path)

        self._fragment_paths = fragment_paths
        return None

    def read_fragment(self, fragment_path: str) -> np.ndarray:
        """
        Read from the given data fragment path.

        Returns a np.ndarray of the ADC waveforms that were read and
        overwrites :self.adc_data: and :self.timestamp_data:.
        """
        if self._verbosity >= 2:
            print("="*60)
            print(f"INFO: Reading from the path\n{fragment_path}")

        fragment = self._h5_file.get_frag(fragment_path)
        fragment_data_size = fragment.get_data_size()

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
            return np.array([])

        wibeth_unpacker = WIBEthUnpacker(self.channel_map, ana_data_prescale=None, wvfm_data_prescale=1)

        _, waveforms = wibeth_unpacker.get_det_data_all(fragment)

        adc_data = np.zeros((len(waveforms), len(waveforms[0].adcs)))
        timestamp_data = waveforms[0].timestamps  # Same for all waveforms
        channel_data = np.zeros((len(waveforms),))

        for idx, waveform in enumerate(waveforms):
            adc_data[idx] = waveform.adcs
            channel_data[idx] = waveform.channel

        self.adc_data = adc_data
        self.timestamp_data = timestamp_data
        self.channel_data = channel_data

        return self.adc_data.copy()

    def clear_data(self) -> None:
        self.adc_data = np.array([])
        self.timestamp_data = np.array([])
        self.channel_data = np.array([])
