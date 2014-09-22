#!/bin/bash

dataname=$1
node=$2
bzip2 -d $dataname.bz2
python ./split-spreadtmp.py /home/jimwang/hostfile_$node $dataname

#scp -r $dataname.$node  jimwang@cn3:/tmp
#scp -r $dataname.$node  jimwang@cn6:/tmp
#scp -r $dataname.$node  jimwang@cn7:/tmp

