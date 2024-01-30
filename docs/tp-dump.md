# Trigger Primitive Dump Info

`tp_dump.py` is a plotting script that shows TP diagnostic information, such as: TP channel histogram and channel vs time over threshold. Plots are saved as SVGs and PNGs.

While running, this script prints various loading information. These outputs can be suppressed with `--quiet`.

One can specify which fragments to load from with the `--start-frag` option. This is -10 by default in order to get the last 10 fragments for the given file. One can also specify which fragment to end on (not inclusive) with `--end-frag` option. This is 0 by default (for the previously mentioned reason).

## Example
```bash
python ta_dump.py file.hdf5 # Loads last 10 fragments by default.
python ta_dump.py file.hdf5 --help
python ta_dump.py file.hdf5 --quiet
python ta_dump.py file.hdf5 --start-frag 50 --end-frag 100 # Loads 50 fragments.
```
