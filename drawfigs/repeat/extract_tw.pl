#!/usr/bin/perl 
open(MYFILE,"$ARGV[0]") || die;
my @timearr;
while(<MYFILE>){
	chomp $_;
	@total = split(/\s+/,$_);
	if( (scalar @total > 3) and ($total[2] eq 'totaltime' ) ){
		print "$total[3]\n";
		push @timearr, $total[3];
	}
}
close MYFILE;


open(MYFILE,"$ARGV[1]") || die;
my @testacc;
my @trainobj;
while(<MYFILE>){
	chomp $_;
	@total = split(/\s+/,$_);
	if( ($total[0] eq 'Accuracy' ) ){
		$len = length($total[2]);
		$tmp = substr $total[2], 0, $len-1;
		print "$tmp\n";
		push @testacc, $tmp;
	}
	if( ($total[0] eq 'primal' ) ){
		print "$total[5]\n";
		push @trainobj, $total[5];
	}
}
close MYFILE;

open(FILE,">$ARGV[2]") || die;
$num = scalar @timearr;
for ($i = 0; $i < $num; $i++) {
 	print FILE "$timearr[$i], $testacc[$i], $trainobj[$i]\n";
}
close FILE;

