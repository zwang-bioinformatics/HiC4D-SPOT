# Author: Bishal Shrestha
# Date: 03-24-2025  

import os
import pandas as pd

def runfindloopsMustache(num_timepoints, chr, resolution, dir_out, model_name):
    """
    wget ftp://cooler.csail.mit.edu/coolers/hg19/Rao2014-GM12878-MboI-allreps-filtered.5kb.cool
    mustache -f ./Rao2014-GM12878-MboI-allreps-filtered.5kb.cool -ch chr12 chr19 -r 5kb -pt 0.05 -o cooler_out.tsv
    OR
    mustache -f ./Rao2014-GM12878-MboI-allreps-filtered.5kb.cool -r 5kb -pt 0.05 -o cooler_out.tsv
    OR
    command = f'python /home/bshrestha/tools/mustache/mustache/mustache.py -f {cool_file} -ch 8 -r 40kb -pt 0.1 -p 10 -o {output_file}'
    """
    # Use mustache to find loops
    print("\n\n\nRunning runfindloopsMustache", flush=True)
    dir_out_mustache = f'{dir_out}/mustache'
    os.makedirs(dir_out_mustache, exist_ok=True)
    # for data_label in ["true", "true_perturbed", "pred", "anomaly", "anomaly_refed"]:
    for data_label in ["anomaly", "anomaly_combined"]:
        for t in range(1, num_timepoints+1):
            print(f"\n\nFinding loops for {data_label} at timepoint {t}", flush=True)
            cool_file = f'{dir_out}/.cool/{model_name}_{data_label}_t{t}.cool'
            
            output_file = f'{dir_out_mustache}/{model_name}_{data_label}_t{t}_loops_mustache.tsv'
            
            # Check if the file with the same prefix exists, if yes, skip
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping...")
                continue
            
            command = f'mustache -f {cool_file} -ch {chr} -r {resolution} -pt 0.01 -p 20 -norm False -o {output_file}'
            os.system(command)

            # Load the file and then sort it on the basis of lowest FDR. The first row are the column name
            df = pd.read_csv(output_file, sep='\t')
            df = df.sort_values(by='FDR')
            df.to_csv(output_file, sep='\t', index=False)


    print(f"runfindloopsMustache completed", flush=True)
            

def runfindloopsChromosight(num_timepoints, dir_out, model_name):
    """
    Usage:
    chromosight detect  [--kernel-config=FILE] [--pattern=loops]
                        [--pearson=auto] [--win-size=auto] [--iterations=auto]
                        [--win-fmt={json,npy}] [--norm={auto,raw,force}]
                        [--subsample=no] [--inter] [--tsvd] [--smooth-trend]
                        [--n-mads=5] [--min-dist=0] [--max-dist=auto]
                        [--no-plotting] [--min-separation=auto] [--dump=DIR]
                        [--threads=1] [--perc-zero=auto]
                        [--perc-undetected=auto] <contact_map> <prefix>
    """
    print("\n\n\nRunning Chromosight")
    dir_out_chromosight = f'{dir_out}/chromosight'
    os.makedirs(dir_out_chromosight, exist_ok=True)
    for data_label in ["true", "true_perturbed", "pred", "anomaly"]:
        for t in range(1, num_timepoints+1):
            print(f"Finding loops for {data_label} at timepoint {t}")
            cool_file = f'{dir_out}/.cool/{model_name}_{data_label}_t{t}.cool'
            output_prefix = f'{dir_out_chromosight}/{model_name}_{data_label}_t{t}_loops_chromosight'
            
            # Check if the file with the same prefix exists, if yes, skip
            if os.path.exists(f'{output_prefix}.tsv'):
                print(f"File {output_prefix}.tsv already exists. Skipping...")
                continue
            
            command = f'chromosight detect --threads 50 {cool_file} {output_prefix}'
            os.system(command)
    print(f"runfindloopsChromosight completed", flush=True)
            

def get_true_loops(chr_list=['chr8'], region=(94_000_000, 96_000_000), development_mode="all", num_loops = 5, loop_loc=''):
    import sys
    import pandas as pd
    import numpy as np

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Example usage
    file_path = loop_loc
    if file_path == '':
        assert False, "Please provide the path to the file containing the loops"
    score_column = 'SIP_APScoreAvg'  # Replace with the column you want to sort by (e.g., SIP_APScoreAvg or SIP_value)
    n = num_loops  # Replace with the number of top rows you want to retrieve

    # Read the data from an Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Make sure it in intra chromosomal: check columns chromosome1 and chromosome2 are same and it needs to be of chromosome chrs_test = ['2', '6', '8']
    df = df[df['chromosome1'] == df['chromosome2']]
    df = df[df['chromosome1'].isin(chr_list)]
    
    # Loops within certain range
    df = df[(df['x1'] >= region[0]) & (df['y2'] <= region[1])]


    time_columns = ['call0000', 'call0030', 'call0060', 'call0090', 'call0120', 'call0240', 'call0360', 'call1440']
    if (development_mode == "all"):
        df = df[df[time_columns].sum(axis=1) == 8]
    elif (development_mode == "gain_late"):
        df = df[(df[time_columns[:3]].sum(axis=1) == 0) & (df[time_columns[3:]].sum(axis=1) > 1)]  # select those where first 4 are 0 and last 4 are 1
        
        
    df_sorted = df.sort_values(by=score_column, ascending=False)
    if type(n) == int:
        top_loops = df_sorted.head(n)
    else:
        top_loops = df_sorted
    
    # Filter out the columns, I only need the chromosome1, start1, end1, chromosome2, start2, end2
    top_loops = top_loops[['chromosome1', 'x1', 'x2', 'chromosome2', 'y1', 'y2']]
    top_loops = top_loops.values.tolist()   # Convert the DataFrame to a list
    
    return top_loops

    # Optionally, save the top n rows to a new Excel file for easy reference
    # top_loops.to_excel('top_loops.xlsx', index=False)