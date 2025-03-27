# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Extract the intra-chromosomal interactions at 10kb resolution using straw and save them in a cooler file using cooler load command or directly use hic2cool
# Select 4 in args_mega before running this script

import os
import sys
import json
import time
import hicstraw
import argparse
import importlib
import numpy as np
import pandas as pd

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
config = importlib.import_module(module_name).get_args()

if config['verbose']: print(f"Time: {time.ctime()}, Process ID: {os.getpid()}, Arguments: {json.dumps(config, indent=4)}", flush=True)


resolution = config["resolution"]
dir_out = config["cooler_dir"]

os.makedirs(dir_out, exist_ok=True)

hic_files = [
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0000_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0030_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0060_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0090_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0120_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0240_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_0360_S_0.0.0_megaMap_inter.hic",
    f"{config['hic_dir']}/GSE201353_LIMA_THP1_WT_LPIF_1440_S_0.0.0_megaMap_inter.hic",
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


# Rename 
map = {
    "0000": "t1",
    "0030": "t2",
    "0060": "t3",
    "0090": "t4",
    "0120": "t5",
    "0240": "t6",
    "0360": "t7",
    "1440": "t8"
}

# only for .cool files
for file in os.listdir(dir_out):
    if file.endswith(".cool"):
        for key in map:
            if key in file:
                new_name = map[key] + ".cool"
                os.rename(dir_out + file, dir_out + new_name)
                print(f"{file} -> {new_name}")
                break

# Command to run this script with nohup and save the output and error in a log file
# nohup python3 Step\ 1\:\ hic2cool.py > Step\ 1\:\ hic2cool.log 2>&1 &