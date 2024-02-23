# Trigger Candidate Dump Info
`tc_dump.py` is a plotting script that shows TC diagnostic information. This includes histograms of all the available data members and ADC integral (sum of all contained TA ADC integrals) and a few light analysis plots:time difference histogram (various start and end time definitions), ADC integral vs number of TAs scatter plot, time spans per TC plot (including a calculation for the number of ticks per TC). These plots are written to a single PDF with multiple pages.

There are two plotting options `--linear` and `--log` that set the y-scale for the plots. By default, plots use both scales with linear on the left y-axis and log on the right y-axis. There is an additional plotting option `--seconds` to produce time plots using seconds instead of ticks.

While running, this prints warnings and general information about loading and plotting. These outputs can be suppressed with `--quiet`.

One can specify which fragments to _attempt_ to load from with the `--start-frag` option. This is `-10` by default in order to get the last 10 fragments for the given file. One can also specify which fragment to end on (not inclusive) with `--end-frag` option. This is `0` by default (for the previously mentioned reason).

A text file named `tc_anomalies_<run_number>-<file_index>.txt` is generated that gives reference statistics for each TA data member and gives a count of data members that are at least 2 sigma and 3 sigma from the mean. One can use `--no-anomaly` to stop this file generation.

## Example
```bash
python tc_dump.py file.hdf5
python tc_dump.py file.hdf5 --help
python tc_dump.py file.hdf5 --quiet
python tc_dump.py file.hdf5 --start-frag 50 --end-frag 100 # Attempts 50 fragments
python tc_dump.py file.hdf5 --no-anomaly
python tc_dump.py file.hdf5 --log
python tc_dump.py file.hdf5 --linear
python tc_dump.py file.hdf5 --seconds
```
