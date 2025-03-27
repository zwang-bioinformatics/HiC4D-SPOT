# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Downsample the Hi-C pairs to mitigate variance in read depth across samples
#		1. Get the 40kb resolution cooler files for all the downsampled Hi-C data (Resolution: 40kb (40000 bp) for each bin)
#		2. Balance the cooler files
# Usage: Change the fin and fout paths to the appropriate paths in your system
# Select 1 in args_mega before running this script

use JSON;

# Run the Python script and capture its JSON output
my $json_text = qx( python ../../../args/args_mega.py );  # or whatever your script is named
my $config = decode_json($json_text);	# Convert the JSON string to a Perl data structure

# my $config_filename = './config.json';
# open(my $fh, '<', $config_filename) or die "Cannot open $config_filename: $!";
# my $json_text = do { local $/; <$fh> };
# close($fh);
# my $config = decode_json($json_text);


my @ids = ("sperm_rep123","MII_rep12","PN5_rep1234","early_2cell_rep123",
           "late_2cell_rep1234","8cell_rep123","ICM_rep123","mESC_500_rep12");

my @ids2 = ("PN5","early_2cell",
           "late_2cell","8cell","ICM","mESC_500");


my $allValidPairs_downsample = $config->{'allValidPairs_downsample_dir'};
my $chrom_size_file = $config->{'chrom_size_file'};


for(my $i=0; $i<@ids2; $i++){
	print "$ids2[$i]\n";
	my $fin = "$allValidPairs_downsample/$ids2[$i].downsample";
	my $fout_dir = $config->{'cooler_dir'};
	my $fout = "$fout_dir/$ids2[$i].cool";

	if (!-d $fout_dir){
		`mkdir $fout_dir`;
	}
	
	print "Running cooler close pairs\n";
	`cooler cload pairs -c1 2 -p1 3 -c2 5 -p2 6 --chunksize 1000000 $chrom_size_file:40000 $fin $fout`;

	# print "Running cooler balance\n\n";
	# `cooler balance -p 20 $fout`;
	# Here -p 20 is the number of threads to use.
}
