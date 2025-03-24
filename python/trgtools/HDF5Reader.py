"""
Generic HDF5Reader class to read and store data.
"""
from hdf5libs import HDF5RawDataFile
from tqdm import tqdm 

import abc


class HDF5Reader(abc.ABC):
    """
    Abstract reader class for HDF5 files.

    Derived classes must complete all methods
    decorated with @abc.abstractmethod.
    """

    # Useful print colors
    _FAIL_TEXT_COLOR = '\033[91m'
    _WARNING_TEXT_COLOR = '\033[93m'
    _BOLD_TEXT = '\033[1m'
    _END_TEXT_COLOR = '\033[0m'


    def __init__(self, filename: str, verbosity: int = 0, batch_mode: bool = False) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            verbosity (int): Verbose level. 0: Only errors. 1: Warnings. 2: All.

        Returns nothing.
        """
        # Generic loading
        self._h5_file = HDF5RawDataFile(filename)
        self._fragment_paths = self._h5_file.get_all_fragment_dataset_paths()
        self.run_id = self._h5_file.get_int_attribute('run_number')
        self.file_index = self._h5_file.get_int_attribute('file_index')

        self._verbosity = verbosity

        self._filter_fragment_paths()  # Derived class must define this.

        self._num_empty = 0  # Counts the number of empty fragments.

        self._batch_mode = batch_mode

        return None

    @abc.abstractmethod
    def _filter_fragment_paths(self) -> None:
        """
        Filter the fragment paths of interest.

        This should be according to the derived reader's
        data type of interest, e.g., filter for TriggerActivity.
        """
        ...

    def get_fragment_paths(self) -> list[str]:
        """ Return the list of fragment paths. """
        return list(self._fragment_paths)

    def set_fragment_paths(self, fragment_paths: list[str]) -> None:
        """ Set the list of fragment paths. """
        self._fragment_paths = fragment_paths
        return None

    @abc.abstractmethod
    def read_fragment(self, fragment_path: str) -> None:
        """ Read one fragment from :fragment_path:. """
        ...

    def read_all_fragments(self) -> None:
        """ Read all fragments. """
        for fragment_path in tqdm(self._fragment_paths, desc='Reading all the fragments', disable=self._batch_mode):
            _ = self.read_fragment(fragment_path)

        # self.read_fragment should increment self._num_empty.
        # Print how many were empty as a debug.
        if self._verbosity >= 1 and self._num_empty != 0:
            print(
                    self._FAIL_TEXT_COLOR
                    + self._BOLD_TEXT
                    + f"WARNING: Skipped {self._num_empty} frags."
                    + self._END_TEXT_COLOR
            )

        return None

    @abc.abstractmethod
    def clear_data(self) -> None:
        """ Clear the contents of the member data. """
        ...

    def reset_fragment_paths(self) -> None:
        """ Reset the fragment paths to the initialized state. """
        self._fragment_paths = self._h5_file.get_all_fragment_dataset_paths()
        self._filter_fragment_paths()
