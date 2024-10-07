# Python trgtools Module

Reading a DUNE-DAQ HDF5 file for the TP, TA, and TC contents can be easily done using the `trgtools` Python module.

# Example

## Common Methods
```python
import trgtools

tpr = trgtools.TPReader(hdf5_file_name)

# Get all the available paths for TPs in this file.
frag_paths = tpr.get_fragment_paths()

# Read all fragment paths. Appends results to tpr.tp_data.
tpr.read_all_fragments()

# Read only one fragment. Return result and append to tpr.tp_data.
frag0_tps = tpr.read_fragment(frag_paths[0])

# Reset tpr.tp_data. Keeps the current fragment paths.
tpr.clear_data()

# Set a different list of fragment paths.
tpr.set_fragment_paths(tpr.get_fragment_paths()[:4])

# Reset the fragment paths to the initalized state.
tpr.reset_fragment_paths()
```

## Data Accessing
```python
tpr = trgtools.TPReader(hdf5_file_name)
tar = trgtools.TAReader(hdf5_file_name)
tcr = trgtools.TCReader(hdf5_file_name)

tpr.read_all_fragments()
tar.read_all_fragments()
tcr.read_all_fragments()

# Primary contents of the fragments
# np.ndarray with each index as one T*
print(tpr.tp_data)
print(tar.ta_data)
print(tcr.tc_data)

# Secondary contents of the fragments
# List with each index as the TPs/TAs in the TA/TC
print(tar.tp_data)
print(tcr.ta_data)

ta0_contents = tar.tp_data[0]
tc0_contents = tcr.ta_data[0]
```
Data accessing follows a very similar procedure between the different readers. The TAReader and TCReader also contain the secondary information about the TPs and TAs that formed the TAs and TCs, respectively. For the `np.ndarray` objects, one can also specify the member data they want to access. For example,
```python
tar.ta_data['time_start']  # Returns a np.ndarray of the time_starts for all read TAs
```
The available data members for each reader can be used (and shown) with `tpr.tp_dt`, `tar.ta_dt` and `tar.tp_dt`, and `tcr.tc_dt` and `tcr.ta_dt`.

Look at the contents of `*_dump.py` for more detailed examples of data member usage.

While using interactive Python, one can do `help(tpr)` and `help(tpr.read_fragment)` for documentation on their usage (and similarly for the other readers and plotters).

One can also access the primary data by indexing the reader class itself. For example,
```python
tar[0]             # Returns the 0-th TA that was read.
tar['time_start']  # Returns the `time_start` values for all TAs read.
len(tar)           # Returns the number of TAs read.
```
It is still necessary to specify the secondary data, e.g., `tar.tp_data[0]` for TPs in the 0-th TA or `tcr.ta_data[10]['adc_integral']` for the TA ADC integrals in the 10-th TC.

# Plotting
There is also a submodule `trgtools.plot` that features a class `PDFPlotter`. This class contains common plotting that was repeated between the `*_dump.py`. Loading this class requires `matplotlib` to be installed, but simply doing `import trgtools` does not have this requirement.

## Example
```python
import trgtools
from trgtools.plot import PDFPlotter

tpr = trgtools.TPReader(file_to_read)

pdf_save_name = 'example.pdf'
pdf_plotter = PDFPlotter(pdf_save_name)

plot_style_dict = dict(title="ADC Peak Histogram", xlabel="ADC Counts", ylabel="Count")
pdf_plotter.plot_histogram(tpr['adc_peak'], plot_style_dict)
```

By design, the `plot_style_dict` requires the keys `title`, `xlabel`, and `ylabel` at a minimum. More options are available to further change the style of the plot, and examples of this are available in the `*_dump.py`.

### Development
The common plots available in `PDFPlotter` is rather limited right now. At this moment, these plots are sufficient, but more common plotting functions can be added.
