# Emulate from TP Stream

`emulate_from_tpstream.cxx` (and the application `trgtools_emulate_from_tpstream`)
processes timeslice HDF5 files that contain TriggerPrimitives, and creates a
new HDF5 that includes TriggerActivities and TriggerCandidates.

The primary use of this is to test TA algorithms, TC algorithms, and their
configurations, with output diagnostics available from `ta_dump.py` and
`tc_dump.py`. The full TC displays that include all TAs & TPs within those TAs
can be made with the `plot_emulated_triggers.pd` script.

The application understands that there might be multiple sources of trigger
primitives, that different files might contain TPs from different sources
(e.g. two APAs), or that different files might contain TPs from the same sources
but across different time-periods (e.g. two consecutive files from the same
APAs).

The application will first check if we have TPs from all the available sources
for the specified slices -- and crop the requested timeslice ranges as
appropriate.

**WARNING:** This script is different from `process_tpstream`, and does not
contain all the functionality yet. Look at TODOs below for more info.

## Example

```bash
trgtools_emulate_from_tpstream -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json --quiet 

trgtools_emulate_from_tpstream -i input_file_APA*.hdf5 -o output_file.hdf5 -j algo_config.json
```

### Algorithm Configuration

An example `algo_config.json` file with an explanation are provided [HERE](README.md#configuration).

### Parallelisation

You can run the emulator with `--parallel`, which will not only process each TPWriter separately, but each TAMaker too. **This is currently only worth trying if using slow algo, like DBSCAN**, otherwise it can be marginally slower.

## TODOs

1. **Latency measurements**: At the moment the script does not calculate latencies for each TP->TA->TC in a way that `convert-tplatencies` does -- and that needs to be corrected.
2. **PDS TP Emulation**: Should be easy to add, but at the moment only works for TPC.
3. **SliceID -> Time error checks**: At the moment the code looks for matching `SliceID`s across different files, and cnverges on a `SliceID` "window" to get the TPs from. Although rare, it should be possible to have matching `SliceID`s with mismatching times -- we need to go off actual TP times, rather than `SliceID`s.
4. **Choose time-slice**: The user currently cannot choose time-slice to plot: the script will match the largest timeslice available across TPWriters provided. That has to be done with combination of the above, so e.g. allow for "time offset" and "time range" to plot for (in ticks, seconds, or whatever)
5. **Parallelisation**: could be optimised further.
