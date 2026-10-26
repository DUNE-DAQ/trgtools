"""
Generic HDF5Reader class to read and store data.
"""
import abc
import warnings

import rawdatafilelibs
from tqdm import tqdm


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"trgtools reader {old} is deprecated; use {new}", DeprecationWarning, stacklevel=3)


class HDF5Reader(abc.ABC):
    """
    Abstract reader class for raw data files (HDF5, or any format rawdatafilelibs reads).

    Fragments are identified by keys: (RecordID, SourceID) pairs. Dataset paths are still
    accepted and returned by the deprecated *_fragment_paths methods and by read_fragment(path).

    Derived classes must complete all methods decorated with @abc.abstractmethod,
    and set _FRAGMENT_TYPE to the fragment type they read (e.g. "Trigger_Activity").
    """
    # Useful print colors
    _FAIL_TEXT_COLOR = '\033[91m'
    _WARNING_TEXT_COLOR = '\033[93m'
    _BOLD_TEXT = '\033[1m'
    _END_TEXT_COLOR = '\033[0m'

    # The type of the fragments this reader reads, as named by daqdataformats; set by derived classes
    _FRAGMENT_TYPE = None

    def __init__(self, filename: str, verbosity: int = 0, batch_mode: bool = False) -> None:
        """
        Loads a given HDF5 file.

        Parameters:
            filename (str): HDF5 file to open.
            verbosity (int): Verbose level. 0: Only errors. 1: Warnings. 2: All.

        Returns nothing.
        """
        # Generic loading
        self._h5_file = rawdatafilelibs.open(filename)
        self.run_id = self._h5_file.metadata.run_number
        self.file_index = self._h5_file.metadata.file_index

        self._verbosity = verbosity

        self._path_to_key = None  # filled on first use of a dataset path (deprecated)
        self.reset_fragment_keys()

        self._num_empty = 0  # Counts the number of empty fragments.

        self._batch_mode = batch_mode

        return None

    def get_fragment_keys(self) -> list:
        """ Return the list of fragment keys, (RecordID, SourceID) pairs. """
        return list(self._fragment_keys)

    def set_fragment_keys(self, fragment_keys: list) -> None:
        """ Set the list of fragment keys. """
        self._fragment_keys = list(fragment_keys)
        return None

    def reset_fragment_keys(self) -> None:
        """ Reset the fragment keys to every fragment of this reader's type, record by record. """
        self._fragment_keys = [
            (record.id, source_id)
            for record in self._h5_file
            for source_id in record.source_ids(fragment_type=self._FRAGMENT_TYPE)
        ]
        return None

    def _fragment(self, fragment_key):
        """ The fragment for a key, or (deprecated) a dataset path. """
        if isinstance(fragment_key, str):
            _deprecated("read_fragment(path)", "read_fragment(key), with keys from get_fragment_keys()")
            fragment_key = self._key_of_path(fragment_key)
        record_id, source_id = fragment_key
        return self._h5_file.reader.get_fragment(record_id, source_id)

    # ---- Deprecated: dataset paths (HDF5 files only) ----

    def _path_of_key(self, fragment_key) -> str:
        import hdf5libs  # noqa: F401 - registers the HDF5 backend's Python type, which has get_dataset_path
        record_id, source_id = fragment_key
        return self._h5_file.backend.get_dataset_path(record_id, source_id)

    def _key_of_path(self, fragment_path: str):
        if self._path_to_key is None:
            self._path_to_key = {
                self._path_of_key((record.id, source_id)): (record.id, source_id)
                for record in self._h5_file
                for source_id in record.source_ids()
            }
        return self._path_to_key[fragment_path]

    def get_fragment_paths(self) -> list[str]:
        """ Return the list of fragment dataset paths (deprecated: use get_fragment_keys). """
        _deprecated("get_fragment_paths()", "get_fragment_keys()")
        return [self._path_of_key(fragment_key) for fragment_key in self._fragment_keys]

    def set_fragment_paths(self, fragment_paths: list[str]) -> None:
        """ Set the list of fragment dataset paths (deprecated: use set_fragment_keys). """
        _deprecated("set_fragment_paths()", "set_fragment_keys()")
        self._fragment_keys = [self._key_of_path(fragment_path) for fragment_path in fragment_paths]
        return None

    def reset_fragment_paths(self) -> None:
        """ Reset the fragment list (deprecated: use reset_fragment_keys). """
        _deprecated("reset_fragment_paths()", "reset_fragment_keys()")
        self.reset_fragment_keys()

    # ----

    @abc.abstractmethod
    def read_fragment(self, fragment_key) -> None:
        """ Read one fragment, given by its key (or, deprecated, its dataset path). """
        ...

    def read_all_fragments(self) -> None:
        """ Read all fragments. """
        for fragment_key in tqdm(self._fragment_keys, desc='Reading all the fragments', disable=self._batch_mode):
            _ = self.read_fragment(fragment_key)

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
