# Emulate TPG from raw ADCs

`tpg_emulator.cxx` (and the application `trgtools_tpg_emulator`)
processes trigger records from HDF5 files that contain raw ADCs inside WIB frames, 
and creates and stores timeslices in a new HDF5 file that includes the emulated TriggerPrimitives.

The primary use of this is to test TPG algorithms, and their
configurations. The timeslices in the output HDF5 files from this application 
can be used as input to the `trgtools_emulate_from_tpstream` application.

## Example

```bash
RAW=np04hd_raw_run026300_0050_dataflow0_datawriter_0_20240520T093907.hdf5
trgtools_tpg_emulator -i $RAW -o output_file.hdf5 -j tpg_config.json
```

### TPG Algorithm Configuration

An example `tpg_config.json` file includes the following items:.

```json
{
  "tpg_config": [
    {
      "AVXFrugalPedestalSubtractProcessor":
      {
        "accum_limit" : 10,
        "pedestals" : []
      },
      "AVXThresholdProcessor":
      {
        "plane0" : 90,
        "plane1" : 90,
        "plane2" : 90
      }
    }
  ],
  "offline_channel_map": [
    "PD2HDTPCChannelMap"
  ]
}
```

## Build 

**WARNING**: To build `trgtools` after adding the TPG emulation, the build order in `sourcecode/dbt-build-order.cmake` was modified as follows; note that `trgtools` was moved after `fdreadoutlibs` with respect to its previous (commented out) location:

```bash
set(build_order "daq-cmake"
                "ers"
                "erskafka"
                "logging"
                "utilities"
                "cmdlib"
                "serialization"
                "okssystem"
                "conffwk"
                "oks"
                "oksdalgen"
                "oksutils"
                "dal"
                "confmodel"
                "appmodel"
                "oksconflibs"
                "dbe"
                "opmonlib"
                "rcif"
                "ipm"
                "iomanager"
                "restcmd"
                "appfwk"
                "hermesmodules"
                "daqconf"
                "listrev"
                "detdataformats"
                "trgdataformats"
                "daqdataformats"
                "detchannelmaps"
                "dfmessages"
                "triggeralgs"
                "timing"
                "timinglibs"
                "hdf5libs"
                #"trgtools"
                "datahandlinglibs"
                "hsilibs"
                "trigger"
                "dfmodules"
                "kafkaopmon"
                # FD packages
                "fddetdataformats"
                "tpglibs"
                "fdreadoutlibs"
                "trgtools"
                "asiolibs"
                "fdreadoutmodules"
                "crtmodules"
                "flxlibs"
                "wibmod"
                "sspmodules"
                "uhallibs"
                "rawdatautils"
                "dpdklibs"
                "daphnemodules"
                "ctbmodules"
                "tdemodules"
                "daqsystemtest"
)
```


## TODOs

1. **Review**: This initial implementation needs reviewing, e.g., the application's 
useage, interfaces to the TA/TC emulation, pedestal initalisation via a 
new json configuration field (see example configuration file above).
2. **Improvements**: Further changes in `trgtools` and `tpgtools` will be needed based on the review comments.
