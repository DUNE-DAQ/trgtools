# Processing TP Streams
`emulate_from_tpstream.cxx` (and the application `trgtools_emulate_from_tpstream`)
processes timeslice HDF5 files that contain TriggerPrimitives, and creates a
new HDF5 that includes TriggerActivities and TriggerCandidates.

The primary use of this is to test TA algorithms, TC algorithms, and their
configurations, with output diagnostics available from `ta_dump.py` and
`tc_dump.py`. 

The application understands that there might be multiple sources of trigger
primitives, that differnet files will might contain TPs from different sources,
or that different files might contain TPs from the same sources but across
different time-periods.

Although the user can select the specific timeslice ranges to process, the
application will first check if we have TPs from all the available sources for
the specified slices -- and crop the requested timeslice ranges as appropriate.

## Example
```bash
trgtools_emulate_from_tpstream -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json -m VDColdboxChannelMap --quiet

trgtools_emulate_from_tpstream --latencies -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json
```