#!/usr/bin/env python


from rich import print

import click
import pandas as pd
import hdf5libs
import daqdataformats
import trgtools
from pathlib import Path

# Static data members initialization
pp = hdf5libs.HDF5PathParameters()

pp.detector_group_type = "Trigger";
pp.detector_group_name = "TPC";
pp.element_name_prefix = "Link";
pp.digits_for_element_number = 5;


lp = hdf5libs.HDF5FileLayoutParameters()
lp.path_params_list = [pp];
lp.record_name_prefix = "TimeSlice";
lp.digits_for_record_number = 6;
lp.digits_for_sequence_number = 0;
lp.record_header_dataset_name = "TimeSliceHeader";

_tp_columns = {
    'version',
    'flag',
    'detid',
    'channel',
    'samples_over_threshold',
    'time_start',
    'samples_to_peak',
    'adc_integral',
    'adc_peak',
}


datasets = {
    'cosmics' : [{
        2:  'tpstream_sim/sim_tpstream_apa0_rop2_0000.csv',
        3:  'tpstream_sim/sim_tpstream_apa0_rop3_0000.csv',
    }],

    'beam_nu' : [{
                2:  'tpstream_sim/beam_numu_tps__apa0_rop2_0000.csv',
                3:  'tpstream_sim/beam_numu_tps__apa0_rop3_0000.csv',
            },
            {
                2:  'tpstream_sim/beam_numu_tps__apa0_rop2_0001.csv',
                3:  'tpstream_sim/beam_numu_tps__apa0_rop3_0001.csv',
            }
        ],
    'radio' : [{
        2:  'tpstream_sim/radio_tps__apa0_rop2_0000.csv',
        3:  'tpstream_sim/radio_tps__apa0_rop3_0000.csv',
    }],
}

@click.command()
@click.argument('dataset', default='beam_nu',
                type=click.Choice(list(datasets.keys())))
@click.option('--run-number', '-r', default=53, show_default=True,
              type=int, help='Run number.')
@click.option('--file-index', '-i', default=0, show_default=True,
              type=int, help='File index.')
def main(dataset, run_number, file_index):

    print(f">>> {lp}")

    dataset_name = dataset

    outfile = f'{dataset_name}.hdf5'
    trb_srcid = daqdataformats.SourceID(daqdataformats.SourceID.kTRBuilder, 999)
    base_elem_id = 1000
    ts_nu = 10



    print(f'[cyan]Opening file {outfile}[/cyan]')
    f = hdf5libs.HDF5RawDataFile(
        outfile,
        run_number,
        file_index,
        Path(__file__).name,
        lp, 
        {}
    )



    print(f"Building tpstream for dataset '{dataset_name}'")
    ts_dfs = [{
        # pd.read_csv("tpstream_sim/sim_tpstream_apa0_rop0_0000.csv"),
        # pd.read_csv("tpstream_sim/sim_tpstream_apa0_rop1_0000.csv"),
        # pd.read_csv("tpstream_sim/sim_tpstream_apa0_rop2_0000.csv"),
        # pd.read_csv("tpstream_sim/sim_tpstream_apa0_rop3_0000.csv"),
        rop:pd.read_csv(f) for rop,f in dataset_entries.items()
    } for dataset_entries in datasets[dataset_name] ]




    for k, rop_dfs in enumerate(ts_dfs):
        
        tsb  = trgtools.TriggerPrimitiveTimeSliceBuilder(ts_nu+k, run_number, trb_srcid)

        print(f'Building timeslice: {tsb.get_timeslice().get_header().timeslice_number}')


        # import IPython; IPython.embed(colors='neutral')
        for i, df in rop_dfs.items():
            df_colset = set(df.columns)
            if not _tp_columns.issubset(df_colset):
                raise RuntimeError(f"Missing columns from dataframe (rop {i}):  {_tp_columns - df_colset}")


        for i, df in rop_dfs.items():
            print(f'Adding rop {i} fragment')
            tsb.add_fragment(
                base_elem_id+i,
                *(df[c].to_numpy(dtype="uint64") for c in _tp_columns)
            )

        print(f'[green]Writing timeslice {tsb.get_timeslice().get_header().timeslice_number}[/green]')
        f.write(tsb.get_timeslice())

    print(f"[green]'{outfile}' completed[/green]")


if __name__ == '__main__':
    main()