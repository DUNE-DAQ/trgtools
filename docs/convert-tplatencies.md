# TP latencies to TA latencies converter script

`convert_tplatencies.py` is a script that takes the huge ta latencies CSV output when running the emulation with `--latencies` enabled, and converts it to a simpler per-TA format. The raw HDF5 output from that emulation run also needs to be provided to the script to extract the TCs. This script will not deal with the TC latencies output. Each row represents one TA created, and the columns represent:

1. Time taken to process the very last TP, which results in a TA being made.
2. Time taken to process all the TPs from TA's `time_start` to the very last TP that results in TA being made (inclusive).
3. Time taken to process all the TPs in TA's window.
4. Mean TP processing time extracted from the TA's window.

The new CSV is saved in the same folder as the large CSV input, with the same name but and extra `_simplified` postfix before the file extension.

There are two compulsory inputs for the script to run:

1. `--latencies` or `-l` is the `.csv` file with the full ta latency output from the `process_tpstream` executable.
2. `--raw` or `-r` is the `.hdf5` output file from the `process_tpstream` executable.

## Example
```bash
python convert_tplatencies.py -r file.hdf5 -l ta_timings_file.csv
```
