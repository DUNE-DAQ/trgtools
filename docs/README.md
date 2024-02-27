# Trigger Emulation & Analysis Tools

The `trgtools` repository contains a collection of tools and scripts to emulate, test and analyze the performance of trigger and trigger algorithms.

Use `pip install -r requirements.txt` to install all the Python packages necessary to run the `*_dump.py` scripts and the `trgtools.plot` submodule.

- `process_tpstream`: Example of a simple pipeline to process TPStream files (slice by slice) and apply a trigger activity algorithm.
- `ta_dump.py`: Script that loads HDF5 files containing trigger activities and plots various diagnostic information. [Documentation](ta-dump.md).
- `tc_dump.py`: Script that loads HDF5 files containing trigger primitives and plots various diagnostic information. [Documentation](tc-dump.md).
- `tp_dump.py`: Script that loads HDF5 files containing trigger primitives and plots various diagnostic information. [Documentation](tp-dump.md).
- Python `trgtools` module: Reading and plotting module in that specializes in reading TP, TA, and TC fragments for a given HDF5. The submodule `trgtools.plot` has a common `PDFPlotter` that is used in the `*_dump.py` scripts. [Documentation](py-trgtools.md).
