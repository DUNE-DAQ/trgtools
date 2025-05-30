# Processing TP Streams

`process_tpstream.cxx` (and the application `trgtools_process_tpstream`) processes a timeslice HDF5 that contains TriggerPrimitives and creates a new HDF5 that also includes TriggerActivities and TriggerCandidates. The primary use of this is to test TA algorithms, TC algorithms, and their configurations, with output diagnostics available from `ta_dump.py` and `tc_dump.py`. The application also outputs the latencies per-tp for TA emulation, and per-TA for TC emulation, into a CSV file, if `--latencies` option is added. The CSV is the format: row as a TP (TA) for TA(TC) emulation, with `time_start`, `adc_integral`, `processing time`, and whether it was TP that closed TA (1) or not (0).

## Example

```bash
trgtools_process_tpstream -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json -m VDColdboxTPCChannelMap --quiet

trgtools_process_tpstream --latencies -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json
```

In the second case, the default map will be `VDColdboxTPCChannelMap`. The `DUNE-DAQ/detchannelmaps` repository has a [table of available channel maps](https://github.com/DUNE-DAQ/detchannelmaps/blob/develop/docs/channel-maps-table.md).

## Algorithm Configuration

An example `algo_config.json` file with an explanation is provided [HERE](README.md#configuration).
