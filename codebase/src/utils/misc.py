# Author: Bishal Shrestha
# Date: 03-24-2025  

import cooler
import cooltools
import seaborn as sns

fruitpunch = sns.blend_palette(['white', 'red'], as_cmap=True)
black_to_red = sns.blend_palette(['black', 'red'], as_cmap=True)

def get_chrom_length(chr, path_to_cool):
    """
    Get the length of the chromosome from the .cool file
    """
    clr = cooler.Cooler(path_to_cool)
    chr_len = clr.chromsizes[chr]
    return chr_len
    
    
    
    