# Processing TP Streams
`process_tpstream.cxx` (and the application `trgtools_process_tpstream`) processes a timeslice HDF5 that contains TriggerPrimitives and creates a new HDF5 that also includes TriggerActivities and TriggerCandidates. The primary use of this is to test TA algorithms, TC algorithms, and their configurations, with output diagnostics available from `ta_dump.py` and `tc_dump.py`. The application also outputs the latencies per-tp for TA emulation, and per-TA for TC emulation, into a CSV file, if `--latencies` option is added. The CSV is the format: row as a TP (TA) for TA(TC) emulation, with `time_start`, `adc_integral`, `processing time`, and whether it was TP that closed TA (1) or not (0).

## Example
```bash
trgtools_process_tpstream -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json -p TriggerActivityMakerExamplePlugin -m VDColdboxChannelMap --quiet

trgtools_process_tpstream --latencies -i input_file.hdf5 -o output_file.hdf5 -j algo_config.json
```
In the second case, the default map will be `VDColdboxChannelMap`.

### Configuration
The `algo_config.json` configuration mirrors the format that is used in `daqconf`. An example is shown below.
```
{
	"trigger_activity_config": [
	    {
		"adc_threshold": 10000,
		"adj_tolerance": 4,
		"adjacency_threshold": 6,
		"n_channels_threshold": 8,
		"prescale": 100,
		"print_tp_info": false,
		"trigger_on_adc": false,
		"trigger_on_adjacency": true,
		"trigger_on_n_channels": false,
		"window_length": 10000
	    }
	],
	"trigger_activity_plugin": [
	    "TriggerActivityMakerPrescalePlugin"
	],
	"trigger_candidate_config": [
	    {
		"adc_threshold": 10000,
		"adj_tolerance": 4,
		"adjacency_threshold": 6,
		"n_channels_threshold": 8,
		"prescale": 100,
		"print_tp_info": false,
		"trigger_on_adc": false,
		"trigger_on_adjacency": true,
		"trigger_on_n_channels": false,
		"window_length": 10000
	    }
	],
	"trigger_candidate_plugin": [
	    "TriggerCandidateMakerPrescalePlugin"
	]
}
```
When developing new algorithms, it is sufficient to change the plugin name and insert the new configurable parameters.
