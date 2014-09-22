#!/bin/bash

#if [ "$#" -lt 13 ]; then
#	echo "Usage: ./runexp.sh [traindata_dir] [testdata_dir] [mpi_hostfile] [np] [regularization_C] [save_record_dor] [name_to_save] [max_iteration] [total_batches] [gamma] [rho_ad] [eta_z] [eta_b]"
#	echo "example:  ./runexp.sh /utmp/a9a.2 /utmp/a9a.t.2 ../hostfile_2 2 1 /utmp/jkrecord  a9a  100  100  0.00003 0.1 1000 13024"
#	exit 1
#fi

traindir=$1
testdir=$2
hostfile=$3
np=$4
c=$5
recorddir=$6
savename=$7
max_iter=$8
total_block=$9
gamma=$10 #1/N  0.00003
rho_ad=$11 #0.1
eta_z=$12  
eta_b=$13

mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/mnist47.$1.tr.txt ./parallel-train -m 1500000  -t 10000 -c $2 -i /utmp/jkrecord/mnist47.psd.interm  /utmp/mnist47_tr.4 

mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/mnist47.$1.ts.txt ./parallel-predict -p /utmp/mnist47_tr.4 /utmp/mnist47_ts.4 /utmp/jkrecord/mnist47.psd.interm


#mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile $3 -np $4 -output-filename $6/$7.tr.txt ./parallel-train -m $8 -t $9 -g $10 -a $11 -z $12 -b $13  -c $5 -i /$6/$7.$np.$c.interm  $1 
#mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile $3 -np $4 -output-filename $6/$7.ts.txt ./parallel-predict -p $1 $2 $6/$7.$np.$c.interm


