# Plot Emulated Triggers

`plot_emulated_triggers.py` is a plotting script that generates one plot for each TriggerCandidate in one PDF. For each TriggerCandidate page, TriggerActivities contained within the TC object are shown in a red box, and TPs contained in each of these TAs are shown as black marker points.

By default, a new PDF is generated (with naming based on the existing PDFs).

While running, this can print information about the file reading using `-v` (warnings) and `-vv` (all). Errors and useful output information (save names and location) are always outputted.

## Example

```bash
python plot_emulated_triggers.py file.hdf5
python plot_emulated_triggers.py file.hdf5 -v
python plot_emulated_triggers.py file.hdf5 -vv
python plot_emulated_triggers.py file.hdf5 --help
```
