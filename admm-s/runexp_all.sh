#!/bin/bash

mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/gamma.icdmf.tr.txt ./parallel-train -f 0.5 -t 1.5 -m 25 -s 1 -c 0.1 -i /utmp/jkrecord/gamma.icdm.interm  /utmp/gamma_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/gamma.icdmf.ts.txt ./parallel-predict -p /utmp/gamma_tr.4 /utmp/gamma_ts.4 /utmp/jkrecord/gamma.icdm.interm

mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/delta.icdmf.tr.txt ./parallel-train -f 0.5 -t 1.5 -m 25 -s 1 -c 0.01 -i /utmp/jkrecord/delta.icdm.interm  /utmp/delta_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/delta.icdmf.ts.txt ./parallel-predict -p /utmp/delta_tr.4 /utmp/delta_ts.4 /utmp/jkrecord/delta.icdm.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/epsilon.icdmf.tr.txt ./parallel-train -f 0.5 -t 1.5 -m 25 -s 1 -c 1 -i /utmp/jkrecord/epsilon.icdm.interm  /utmp/epsilon_normalized.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/epsilon.icdmf.ts.txt ./parallel-predict -p /utmp/epsilon_normalized.4 /utmp/epsilon_normalized.t.4 /utmp/jkrecord/epsilon.icdm.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/mnist47.icdmf.tr.txt ./parallel-train -f 0.5 -t 1.5 -m 25 -s 1 -c 1 -i /utmp/jkrecord/mnist47.icdm.interm  /utmp/mnist47_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/mnist47.icdmf.ts.txt ./parallel-predict -p /utmp/mnist47_tr.4 /utmp/mnist47_ts.4 /utmp/jkrecord/mnist47.icdm.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/rcv1.icdmf.tr.txt ./parallel-train -f 0.5 -t 1.5 -m 25 -s 1 -c 1 -i /utmp/jkrecord/rcv1.icdm.interm  /utmp/rcv1_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/rcv1.icdmf.ts.txt ./parallel-predict -p /utmp/rcv1_tr.4 /utmp/rcv1_ts.4 /utmp/jkrecord/rcv1.icdm.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/ocr.icdmf.tr.txt ./parallel-train -f 0.5 -t 1.5 -m 25 -s 1 -c 1 -i /utmp/jkrecord/ocr.icdm.interm  /utmp/ocr_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/ocr.icdmf.ts.txt ./parallel-predict -p /utmp/ocr_tr.4 /utmp/ocr_ts.4 /utmp/jkrecord/ocr.icdm.interm


