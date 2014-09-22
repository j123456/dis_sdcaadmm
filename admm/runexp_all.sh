#!/bin/bash

mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/gamma_fori.tr.txt ./parallel-train -m 25 -s 1 -c 0.1 -i /utmp/jkrecord/gamma_ori.4.1.interm  /utmp/gamma_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/gamma_fori.ts.txt ./parallel-predict -p /utmp/gamma_tr.4 /utmp/gamma_ts.4 /utmp/jkrecord/gamma_ori.4.1.interm

mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/delta_fori.tr.txt ./parallel-train -m 25 -s 1 -c 0.01 -i /utmp/jkrecord/delta_ori.4.1.interm  /utmp/delta_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/delta_fori.ts.txt ./parallel-predict -p /utmp/delta_tr.4 /utmp/delta_ts.4 /utmp/jkrecord/delta_ori.4.1.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/epsilon_fori.tr.txt ./parallel-train -m 25 -s 1 -c 1 -i /utmp/jkrecord/epsilon_ori.4.1.interm  /utmp/epsilon_normalized.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/epsilon_fori.ts.txt ./parallel-predict -p /utmp/epsilon_normalized.4 /utmp/epsilon_normalized.t.4 /utmp/jkrecord/epsilon_ori.4.1.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/mnist47_fori.tr.txt ./parallel-train -m 25 -s 1 -c 1 -i /utmp/jkrecord/mnist47_ori.4.1.interm /utmp/mnist47_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/mnist47_fori.ts.txt ./parallel-predict -p /utmp/mnist47_tr.4 /utmp/mnist47_ts.4 /utmp/jkrecord/mnist47_ori.4.1.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/cov_fori.tr.txt ./parallel-train -m 25 -s 1 -c 1 -i /utmp/jkrecord/cov_ori.4.1.interm  /utmp/covtype_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/cov_fori.ts.txt ./parallel-predict -p /utmp/covtype_tr.4 /utmp/covtype_ts.4 /utmp/jkrecord/cov_ori.4.1.interm


mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/ocr_fori.tr.txt ./parallel-train -m 25 -s 1 -c 1 -i /utmp/jkrecord/ocr_ori.4.1.interm  /utmp/ocr_tr.4
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/ocr_fori.ts.txt ./parallel-predict -p /utmp/ocr_tr.4 /utmp/ocr_ts.4 /utmp/jkrecord/ocr_ori.4.1.interm


