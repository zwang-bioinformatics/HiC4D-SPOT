# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Downsample the Hi-C pairs to mitigate variance in read depth across samples
#	Two step process on downsampling:
#		1. Extract long-range intra-chromosomal interactions (> 20kb) from allValidPairs
#		2. Downsample to 115M reads
# Usage: Change the fin and dirout base paths to the appropriate paths in your system
# Package installation: conda install -c conda-forge perl-json
# Select 1 in args_mega before running this script

#!/usr/bin/env perl

use warnings;
use strict;
use POSIX;
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


my @ids2 = ("sperm","MII","PN5","early_2cell",
           "late_2cell","8cell","ICM","mESC_500");

my $allValidPairs_dir = $config->{'allValidPairs_dir'};
my $dirout = $config->{'allValidPairs_downsample_dir'};

if (!-d $dirout){
  `mkdir $dirout`;
}

# my $cid = $ARGV[0];	

for(my $i=0; $i<@ids; $i++){
	if ($ids2[$i] eq 'sperm' || $ids2[$i] eq 'MII'){
    next;
  }

	# Here, we are checking if the chromosome ID matches the one passed as an argument
	# if($ids2[$i] ne $cid){
	# 	next;
	# }

  print "$ids2[$i]\n";
  
	my $fin = "$allValidPairs_dir/GSE82185_$ids[$i]_allValidPairs.txt";
	my $fout1 = "$dirout/$ids2[$i].longRange_intra";	# > 20kb
	my $fout2 = "$dirout/$ids2[$i].downsample";			# to 115M

	my $n = 0;
	open OUT, ">$fout1";
	open IN, "$fin";
	while(my $line = <IN>){
  	chomp $line; $line =~ s/^\s+//; $line =~ s/\s+$//;
  	my @items = split(/\s+/, $line);
		if($items[1] eq $items[4] && abs($items[2]-$items[5]) > 20000){
			$n++;
			print OUT "$line\n";
		}

	}close IN;
	close OUT;
	print "$n\n";

	`python random_sampler.py $fout1 115000000 >$fout2`;

}