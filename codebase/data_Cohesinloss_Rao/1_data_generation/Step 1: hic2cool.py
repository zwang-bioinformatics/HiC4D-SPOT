# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Extract the intra-chromosomal interactions at 10KB resolution and save them in a cooler file
# Select 7 in args_mega before running this script

import os
import sys
import argparse
import hicstraw
import importlib
import numpy as np
import pandas as pd

print(f"Process ID: {os.getpid()}", flush=True)

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
config = importlib.import_module(module_name).get_args()

resolution = config["resolution"]
dir_out = config['cooler_dir']

os.makedirs(dir_out, exist_ok=True)

hic_files = [
   f"{config['hic_dir']}/GSE104333_Rao-2017-untreated_combined.hic",
   f"{config['hic_dir']}/GSE104333_Rao-2017-treated_6hr_combined.hic",
   f"{config['hic_dir']}/GSE104333_Rao-2017-treated_20min_withdraw_combined.hic",
   f"{config['hic_dir']}/GSE104333_Rao-2017-treated_40min_withdraw_combined.hic",
   f"{config['hic_dir']}/GSE104333_Rao-2017-treated_60min_withdraw_combined.hic",
   f"{config['hic_dir']}/GSE104333_Rao-2017-treated_180min_withdraw_combined.hic",
]

# Using straw to extract the intra-chromosomal interactions at 10kb resolution and save them in a cooler file
for hic_file in hic_files:
    print(f"Processing {hic_file}", flush=True)
    
    cool_file = hic_file.split('/')[-1].replace('.hic', '.cool')
    cool_file = os.path.join(dir_out, cool_file)

    # command for hic2cool
    command = f'hic2cool convert {hic_file} {cool_file} -r {resolution} -p 50'
    os.system(command)
    
    # balance:
    command = f'cooler balance --max-iters 500 -p 50 {cool_file}'
    os.system(command)

print(f"Done with all files", flush=True)


# Rename 
print("Renaming files", flush=True)
map = {
    "GSE104333_Rao-2017-untreated_combined": "t1",
    "treated_6hr_combined": "t2",
    "treated_20min_withdraw_combined": "t3",
    "treated_40min_withdraw_combined": "t4",
    "treated_60min_withdraw_combined": "t5",
    "treated_180min_withdraw_combined": "t6",
}

# only for .cool files
for file in os.listdir(dir_out):
    if file.endswith(".cool"):
        for key in map:
            if key in file:
                new_name = map[key] + ".cool"
                os.rename(dir_out + file, dir_out + new_name)
                print(f"{file} -> {new_name}")

# Command to run this script with nohup and save the output and error in a log file
# nohup python3 Step\ 1\:\ hic2cool.py > Step\ 1\:\ hic2cool.log 2>&1 &

# ps aux | grep hic2cool | awk '{print $2}' | xargs kill -9
