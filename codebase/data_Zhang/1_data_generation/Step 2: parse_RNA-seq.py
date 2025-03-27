# Author: Bishal Shrestha
# Date: 03-24-2025  
# Parse the RNA seq file, required for plotting
# Select 6 in args_mega before running this script

import os
import sys
import argparse
import importlib
import pandas as pd
from gtfparse import read_gtf

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
config = importlib.import_module(module_name).get_args()

# get the .rpkm files from the folder
dir = config['rna_seq_dir']
dir_out = config['rna_seq_parsed_dir']

gtf_file = config['gtf_file']

os.makedirs(dir_out, exist_ok=True)

files = [f for f in os.listdir(dir) if f.endswith(".rpkm")]

# If the file name contains the key as name anywhere, the new name will be the value with same extension
file_map = {
    'D0_1': 't1',
    'D2_1': 't2',
    'D5_1': 't3',
    'D7_1': 't4',
    'D15_1': 't5',
    'D80_1': 't6',
}

def exons_level():
    for file in files:
        
        # First check if the file name contains the key as name anywhere, if not then skip the file
        if not any(key in file for key in file_map.keys()):
            continue

        # Read the .rpkm file
        df = pd.read_csv(f"{dir}/{file}", sep="\t")

        # Parse the multiple entries by splitting the ';' values
        parsed_data = []
        for index, row in df.iterrows():
            gene = row['Geneid']
            chrs = row['Chr'].split(';')
            starts = row['Start'].split(';')
            ends = row['End'].split(';')
            strand = row['Strand'].split(';')
            length = row['Length']
            rpkm = row[df.columns[-1]]  # Assuming the last column is RPKM value

            for chr_, start, end, strand_ in zip(chrs, starts, ends, strand):
                parsed_data.append([gene, chr_, int(start), int(end), strand_, length, rpkm])

        # Create a DataFrame for easy analysis
        parsed_df = pd.DataFrame(parsed_data, columns=['Geneid', 'Chr', 'Start', 'End', 'Strand', 'Length', 'RPKM'])

        # Save the parsed data to a file with the new name based on the original file name with the help of file_map
        new_file_name = file
        for key, value in file_map.items():
            if key in file:
                new_file_name = value + ".rpkm"
                break
        
        parsed_df.to_csv(f"{dir_out}/{new_file_name}", index=False, sep="\t")
        print(f"File: {file} has been parsed and saved as {new_file_name}")
        
def genes_level():
    gtf = read_gtf(gtf_file)
    genes_gtf = gtf[gtf["feature"] == "gene"]
    genes_gtf = genes_gtf[["seqname", "start", "end", "strand", "gene_name"]].rename(
        columns={"seqname": "Chr"}
    )
    print(genes_gtf.head())
    # Ensure the gene information is consolidated (each gene_id appears only once)
    consolidated_gtf = genes_gtf.groupby("gene_name").agg({
        "Chr": "first",
        "start": "min",
        "end": "max",
        "strand": "first"
    }).reset_index()
    
    for file in files:
        # First check if the file name contains the key as name anywhere, if not then skip the file
        if not any(key in file for key in file_map.keys()):
            continue
        
        # Read the .rpkm file
        rpkm_df = pd.read_csv(f"{dir}/{file}", sep="\t")
        
        rpkm_df.columns.values[-1] = "RPKM"
        
        # Merge the RNA-seq data with gene annotations using gene_id
        merged_df = pd.merge(rpkm_df, consolidated_gtf, left_on="Geneid", right_on="gene_name", how="left")
        
        # Drop one of the redundant 'Chr' columns and rename for consistency
        merged_df = merged_df.drop(columns=["Chr_x"])
        merged_df = merged_df.rename(columns={"Chr_y": "Chr"})
        
        
        merged_df = merged_df[[
            "Geneid", "Chr", "start", "end", "strand", "Length", "RPKM"
        ]]
        print(merged_df.head())
        
        # Save the parsed data to a file with the new name based on the original file name with the help of file_map
        new_file_name = file
        for key, value in file_map.items():
            if key in file:
                new_file_name = value + ".rpkm"
                break
    
        merged_df.to_csv(f"{dir_out}/{new_file_name}", index=False, sep="\t")
        
        print(f"File: {file} has been parsed and saved as {new_file_name}")
        
genes_level()
        
            
    