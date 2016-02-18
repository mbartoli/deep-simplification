#! /usr/bin/perl -w

use strict;

&main();

sub main{
	if( scalar(@ARGV) != 1 ){
		&print_usage();
		exit 1;
	}

	my $parsed_file = $ARGV[0];

	open IN, $parsed_file or die "Problems opening file: $parsed_file\n$!";

	while( my $line = <IN> ){
		chomp $line;

		if( $line =~ /\t/ ){
			my @parts = split /\t/, $line;

			print join("\t", $parts[0], $parts[1], &get_text($parts[2])) . "\n";
		}else{
			print &get_text($line) . "\n";
		}
	}
}

sub get_text{
	my ($line) = @_;

	$line =~ s/\([^ ]*/ /g;
	$line =~ s/\)/ /g;
	$line =~ s/\(/ /g;
	$line =~ s/\s+/ /g;
	$line =~ /^\s*(.*?)\s*$/;

	return $1;
}

sub print_usage{
	print STDERR "get_text_from_parse.pl <parsed_file>\n";
}
