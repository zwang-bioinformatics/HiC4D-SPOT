# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Script to generate plot for the Hi-C matrices containing, True Hi-C, Hi-C with anomalies if any (else True Hi-C again), the detected anomalies.
# In args_mega, select appropriate number before running the script. More detail on args_mega.py

import os
import json
import math
import argparse
import importlib
import pandas as pd

from utils.misc import *
from utils.tads import *
from utils.loops import *
from utils.tracks import *
from utils.plots2 import *
from utils.anomaly import *
from utils.save_hic import *
from utils.results_stats import *
from utils.combine_sub_matrices import *

from configs.study_regions import *

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
args = importlib.import_module(module_name).get_args()

print(f"Process ID: {os.getpid()}")
print("Arguments: ", json.dumps(args, indent=4), flush=True)

model_name = "best"
chrid = args['chr_prefix'] + args['predict_chr']
sub_mat_n = args['sub_matrix_size']      # Sub-matrix/Window size 
resolution = args['resolution'] 
num_timepoints = args['num_timepoints']
# start = args['start']
# end = args['end']

if args['ids']:
    ids = args['ids']
    if args['verbose']: print(f"Using provided IDs: {ids}", flush=True)
    assert isinstance(ids, list), f"IDs should be a list, not {type(ids)}"
else:
    ids = [f"t{i}" for i in range(1, args['num_timepoints']+1)]
    if args['verbose']: print(f"Generated IDs: {ids}", flush=True)

# region = (chrid, start, end)
# region_bin = (chrid, start//resolution, end//resolution)
# if args['verbose']: print("Region: ", region, "Region_bin: ", region_bin)

dir_out = os.path.join(args['output_eval_file'], f"{resolution}bp_{chrid}")
if not os.path.exists(dir_out): os.makedirs(dir_out)

# Get the length (in bp) and number of bins in the chromosome
chr_len = get_chrom_length(chr=chrid, path_to_cool=os.path.join(args['cooler_dir'], f"{ids[0]}.cool"))
num_bins = math.ceil(chr_len/resolution)
if args['verbose']: print("Chromosome length: ", chr_len, "Number of bins: ", num_bins)

# Location of the Hi-C matrix
fmat_true           = os.path.join(args['default_input_data'], f"data_{resolution}bp_{chrid}.npy")
find_true           = os.path.join(args['default_input_data'], f"data_{resolution}bp_{chrid}_index.npy")

fmat_true_perturbed = os.path.join(args['input_data'], f"data_{resolution}bp_{chrid}.npy")
find_perturbed      = os.path.join(args['input_data'], f"data_{resolution}bp_{chrid}_index.npy")

fmat_pred = os.path.join(args['output_predict_dir'], f"{chrid}.npy")


if not args['memory_efficient']:
    
    # Load the HiC matrix
    hics_true           = get_combined_hic(file_loc=fmat_true, file_index_loc=find_true, num_timepoints=num_timepoints, num_bins=num_bins, sub_mat_n=sub_mat_n, dir_out=dir_out, file_name="true", verbose=args['verbose'])
    
    hics_true_perturbed2 = get_combined_hic(file_loc=fmat_true_perturbed, file_index_loc=find_perturbed, num_timepoints=num_timepoints, num_bins=num_bins, sub_mat_n=sub_mat_n, dir_out=dir_out, file_name="true_perturbed", verbose=args['verbose'])
    hics_true_perturbed = hics_true_perturbed2.copy()
    print(f"Shape of hics_true_perturbed: {hics_true_perturbed.shape}", flush=True)
    if args['data_type'] == 'time_swap':
        hics_true_perturbed = np.clip(hics_true_perturbed, -1, 1)
        temp = hics_true_perturbed[1].copy()
        hics_true_perturbed[1] = hics_true_perturbed[5].copy()
        hics_true_perturbed[5] = temp
   
    hics_pred           = get_combined_hic(file_loc=fmat_pred, file_index_loc=find_true, num_timepoints=num_timepoints, num_bins=num_bins, sub_mat_n=sub_mat_n, dir_out=dir_out, file_name="pred", verbose=args['verbose'])
    hics_pred = np.clip(hics_pred, -1, 1)
    
    hics_ano            = get_anomaly_hic(hics_true_perturbed, hics_pred, dir_out, file_name="anomaly", verbose=args['verbose'])
    hics_ano_refed      = get_anomaly_hic_refed(hics_ano, dir_out, file_name="anomaly_refed", verbose=args['verbose'])
    
    hic_ano_combined_array = []
    hic_ano_combined = np.zeros((num_bins, num_bins))
    for i in range(len(hics_ano)):
        hic_ano_combined += hics_ano[i]
    hic_ano_combined = hic_ano_combined / len(hics_ano)
    for i in range(num_timepoints):
        hic_ano_combined_array.append(hic_ano_combined)
        
    # Generate Statistics
    if args['data_type'] == 'time_swap' or args['data_type'] == 'default':
        generate_stats(hics_true, hics_true_perturbed, hics_pred, hics_ano, hics_ano_refed, hic_ano_combined_array, dir_out, model_name, filter_zeros = False, data_type=args['data_type'], verbose=args['verbose'])

    # Save the HiC matrix in .cool format
    save_hic_as_cool(hics_true, num_bins=num_bins, bin_size=resolution, chromosome=chrid, dir_out=dir_out, output_filename=f"{model_name}_true", assembly=args['assembly'], verbose=args['verbose'])
    save_hic_as_cool(hics_true_perturbed, num_bins=num_bins, bin_size=resolution, chromosome=chrid, dir_out=dir_out, output_filename=f"{model_name}_true_perturbed", assembly=args['assembly'], verbose=args['verbose'])
    save_hic_as_cool(hics_pred, num_bins=num_bins, bin_size=resolution, chromosome=chrid, dir_out=dir_out, output_filename=f"{model_name}_pred", assembly=args['assembly'], verbose=args['verbose'])
    save_hic_as_cool(hics_ano, num_bins=num_bins, bin_size=resolution, chromosome=chrid, dir_out=dir_out, output_filename=f"{model_name}_anomaly", assembly=args['assembly'], verbose=args['verbose'])
    save_hic_as_cool(hic_ano_combined_array, num_bins=num_bins, bin_size=resolution, chromosome=chrid, dir_out=dir_out, output_filename=f"{model_name}_anomaly_combined", assembly=args['assembly'], verbose=args['verbose'])

    
    if args['eval_tads']:
        # Save the HiC matrix in .matrix format
        save_hic_matrix(hics_true, dir_out=dir_out, file_name=f"{model_name}_true", verbose=args['verbose'])
        save_hic_matrix(hics_true_perturbed, dir_out=dir_out, file_name=f"{model_name}_true_perturbed", verbose=args['verbose'])
        save_hic_matrix(hics_pred, dir_out=dir_out, file_name=f"{model_name}_pred", verbose=args['verbose'])
        save_hic_matrix(hics_ano, dir_out=dir_out, file_name=f"{model_name}_anomaly", verbose=args['verbose'])
        save_hic_matrix(hic_ano_combined_array, dir_out=dir_out, file_name=f"{model_name}_anomaly_combined", verbose=args['verbose'])
        
        # convert cool -> h5
        save_cool_as_h5(num_timepoints=num_timepoints, dir_out=dir_out, output_filename=f"{model_name}_true", verbose=args['verbose'])
        save_cool_as_h5(num_timepoints=num_timepoints, dir_out=dir_out, output_filename=f"{model_name}_true_perturbed", verbose=args['verbose'])
        save_cool_as_h5(num_timepoints=num_timepoints, dir_out=dir_out, output_filename=f"{model_name}_pred", verbose=args['verbose'])
        save_cool_as_h5(num_timepoints=num_timepoints, dir_out=dir_out, output_filename=f"{model_name}_anomaly", verbose=args['verbose'])
        save_cool_as_h5(num_timepoints=num_timepoints, dir_out=dir_out, output_filename=f"{model_name}_anomaly_combined", verbose=args['verbose'])

        
        # requires .matrix
        runOnTAD(num_timepoints=num_timepoints, chr=chrid, chr_len=chr_len, resolution=resolution, dir_out=dir_out, output_filename=model_name)
        
        # requires .h5
        runhicFindTADs(num_timepoints=num_timepoints, resolution=resolution, dir_out=dir_out, model_name=model_name)

    if args['eval_loops']:
        # requires .cool
        runfindloopsMustache(num_timepoints=num_timepoints, chr=chrid, resolution=resolution, dir_out=dir_out, model_name=model_name)
        
        # requires .cool
        runfindloopsChromosight(num_timepoints=num_timepoints, dir_out=dir_out, model_name=model_name)


    
###########
# Load the RNA-seq data
###########
rna_seq = []
if args.get("plot_rna", False):
    for i in range(num_timepoints):
        # load the pandas dataframe
        rna_seq.append(pd.read_csv(os.path.join(args['rna_seq_parsed_dir'], f"t{i+1}.rpkm"), sep="\t"))

###########
# Variable for visualization
###########
print("Visualizing Hi-C matrices...", flush=True)
viz_regions = study_regions[args['data']]['regions'][chrid]
    
###########
# Plot the Hi-C matrices for the specified regions
###########
for idx, loop_region in enumerate(viz_regions):
    print(f"\n\nGenerating plots for region: {loop_region}", flush=True)
    region = (chrid, loop_region[0], loop_region[1])
    region_bin = (chrid, region[1]//resolution, region[2]//resolution)
    
    if args['memory_efficient']:
        hics_true           = get_combined_hic_memory_efficient(file_loc=fmat_true, file_index_loc=find_true, num_timepoints=num_timepoints, num_bins=num_bins, sub_mat_n=sub_mat_n, dir_out=dir_out, file_name="true", verbose=args['verbose'], region_bin=region_bin)
        hics_true_perturbed = get_combined_hic_memory_efficient(file_loc=fmat_true_perturbed, file_index_loc=find_perturbed, num_timepoints=num_timepoints, num_bins=num_bins, sub_mat_n=sub_mat_n, dir_out=dir_out, file_name="true_perturbed", verbose=args['verbose'], region_bin=region_bin)
        hics_pred           = get_combined_hic_memory_efficient(file_loc=fmat_pred, file_index_loc=find_true, num_timepoints=num_timepoints, num_bins=num_bins, sub_mat_n=sub_mat_n, dir_out=dir_out, file_name="pred", verbose=args['verbose'], region_bin=region_bin)
        hics_pred = np.clip(hics_pred, -1, 1)
        hics_ano            = get_anomaly_hic(hics_true_perturbed, hics_pred, dir_out, file_name="anomaly", verbose=args['verbose'])
        hics_ano_refed      = get_anomaly_hic_refed(hics_ano, dir_out, file_name="anomaly_refed", verbose=args['verbose'])
        
        # update the region since hics are loaded for the region, not the whole chromosome
        region_bin = (chrid, 0, hics_true[0].shape[0])

    if args['plot_hictracks']:
        hic_track(ids=ids, chr=chrid, region=region, dir_out=dir_out, output_filename=model_name)
        combine_track(region, dir_out=dir_out, output_filename=model_name)

    if args['plot_triangular']:
        show_plot2(hics_true, hics_true_perturbed, hics_pred, hics_ano_refed, rna_seq, region=region, region_bin=region_bin, resolution=resolution, ids=ids, dir_out=dir_out, model_name=model_name, maxV=100, show_loops=False, trianglePlot=True, verbose=args['verbose'], prefix=f"{idx}_")


###########
print("Done!", flush=True)
###########
