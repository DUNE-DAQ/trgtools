# Trigger Activity Dump Info

`ta_dump.py` is a plotting script that shows TA diagnostic information, such as: algorithms produced, number of TPs per TA, event displays, window length histogram, ADC integral histogram, and a plot of the time starts. Most of these plots are saved as an SVG. The event displays are plotted in a multi-page PDF.

While running, this prints warnings for empty fragments that are skipped in the given HDF5 file. These outputs can be suppressed with `--quiet`.

One can specify which fragments to _attempt_ to load from with the `--start-frag` option. This is `-10` by default in order to get the last 10 fragments for the given file. One can also specify which fragment to end on (not inclusive) with `--end-frag` option. This is `0` by default (for the previously mentioned reason).

Event displays are processed by default. If there are many TAs that were loaded, then this may take a while to plot. The `--no-display` options skips event display plotting.

A text file named `ta_anomaly_summary.txt` is generated that gives reference statistics for each TA data member and gives a count of data members that are at least 2 sigma and 3 sigma from the mean. One can use `--no-anomaly` to stop this file generation.

## Example
```bash
python ta_dump.py file.hdf5
python ta_dump.py file.hdf5 --help
python ta_dump.py file.hdf5 --quiet
python ta_dump.py file.hdf5 --start-frag 50 --end-frag 100 # Attempts 50 fragments
python ta_dump.py file.hdf5 --no-display
python ta_dump.py file.hdf5 --no-anomaly
```

## Run Numbers & File Naming
For the moment, getting meaningful run numbers and sub-run numbers is read from the filename because accessing HDF5 run number and sub-run numbers is not available to python. [This](https://github.com/DUNE-DAQ/hdf5libs/pull/68) PR aims to solve that.
