# Trigger Activity Dump Info

`ta_dump.py` is a plotting script that shows TA diagnostic information, such as: algorithms produced, number of TPs per TA, event displays, window length histogram, ADC integral histogram, and a plot of the time starts. Most of these plots are saved as an SVG. The event displays are plotted in a multi-page PDF.

While running, this prints warnings for empty fragments that are skipped in the given HDF5 file. These outputs can be suppressed with `--quiet`.

## Example
```
python ta_dump.py file.hdf5
python ta_dump.py file.hdf5 --quiet
```

## Run Numbers & File Naming
For the moment, getting meaningful run numbers and sub-run numbers is read from the filename because accessing HDF5 run number and sub-run numbers is not available to python. [This](https://github.com/DUNE-DAQ/hdf5libs/pull/68) PR aims to solve that.
