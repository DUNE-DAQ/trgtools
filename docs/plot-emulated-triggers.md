# Plot Emulated Triggers

`plot_emulated_triggers.py` is a plotting script that generates one plot for each TriggerCandidate in one PDF. For each TriggerCandidate page, TriggerActivities contained within the TC object are shown as red boxes. The blue vertical lines show the TriggerPrimitives associated with each TA: the x-position is the TP channel, and the line spans the TP time window from `time_start` to `time_start + 32 * samples_over_threshold`. The black `x` marker shows the TA peak position, using `channel_peak` and `time_peak`.

By default, a new PDF is generated (with naming based on the existing PDFs).

While running, this can print information about the file reading using `-v` (warnings) and `-vv` (all). Errors and useful output information (save names and location) are always outputted.

## Example

```bash
python plot_emulated_triggers.py file.hdf5
python plot_emulated_triggers.py file.hdf5 -v
python plot_emulated_triggers.py file.hdf5 -vv
python plot_emulated_triggers.py file.hdf5 --help
```
