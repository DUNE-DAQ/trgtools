# Trigger Emulation & Analysis Tools

The `trgtools` repository contains a collection of tools and scripts to emulate, test and analyze the performance of trigger and trigger algorithms.

Use `pip install -r requirements.txt` to install all the Python packages necessary to run the `*_dump.py` scripts and the `trgtools.plot` submodule.

- `emulate_from_tpstream`: Prototype of a full pipeline to process TPStream files and apply trigger algorithms (multiple planes & APA/CRPs, no latency measurements yet) [Documentation](emulate-from-tpstream.md).
- `plot_emulated_triggers`: Script that loads HDF5 file generated from `emulate_from_tpstream` and plots TriggerCandidates, one per page. For each TriggerCandidate it plots each TriggerActivity that is contained in the TC, and all the TPs contained within the TA objects. [Documentation](plot-emulated-triggers.md)
- `process_tpstream`: Example of a simple pipeline to process TPStream files (slice by slice) and apply a trigger algorithms (single plane, inc. latency measurements). [Documentation](process-tpstream.md).
- `ta_dump.py`: Script that loads HDF5 files containing trigger activities and plots various diagnostic information. [Documentation](ta-dump.md).
- `tc_dump.py`: Script that loads HDF5 files containing trigger primitives and plots various diagnostic information. [Documentation](tc-dump.md).
- `tp_dump.py`: Script that loads HDF5 files containing trigger primitives and plots various diagnostic information. [Documentation](tp-dump.md).
- `convert_tplatencies.py`: Script that loads HDF5 files and CSV `ta_timings_` files from `process_tpstream` with `--latencies` enabled, and outputs simplified CSV latencies per-TA. [Documentation](convert-tplatencies.md).
- Python `trgtools` module: Reading and plotting module in that specializes in reading TP, TA, and TC fragments for a given HDF5. The submodule `trgtools.plot` has a common `PDFPlotter` that is used in the `*_dump.py` scripts. [Documentation](py-trgtools.md).

## Configuration

Both `emulate_from_tpstream` and `process_tpstream` require a `.json` file with the trigger algorithm configuration, which includes one `TAMaker` and one `TCMaker`. 

The `algo_config.json` configuration below mirrors the format that is used in v4 `daqconf` and the current trigger algorithm setup (it was not ported to OKS due to compatibilities with the offline software). An example is shown below.

```json
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
      "window_length": 10000,
      "max_time_over_threshold": 10000
    }
  ],
  "trigger_activity_plugin": [
    "TAMakerPrescaleAlgorithm"
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
      "window_length": 10000,
      "tc_type_name": "kPrescale"
    }
  ],
  "trigger_candidate_plugin": [
    "TCMakerPrescaleAlgorithm"
  ]
}
```

When developing new algorithms, it is sufficient to change the plugin name and insert the new configurable parameters.
