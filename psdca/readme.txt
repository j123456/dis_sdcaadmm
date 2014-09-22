These codes are modified from the code of
Efficient Distributed Linear Classification Algorithms via the Alternating Direction Method of Multipliers.
Caoxie Zhang, Honglak Lee, and Kang G. Shin
I use it for I/O
##################################################
To run distributed SDCA, please see the sripts.

For example:

#training
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/gamma.$1.tr.txt ./parallel-train -m 4000000  -t 4000 -c $2 -i /utmp/jkrecord/gamma.psd.interm  /utmp/gamma_tr.4 

#testing
mpirun --mca btl_tcp_if_include 192.168.160.0/24 --hostfile ../hostfile_4 -np 4 -output-filename /utmp/jkrecord/gamma.$1.ts.txt ./parallel-predict -p /utmp/gamma_tr.4 /utmp/gamma_ts.4 /utmp/jkrecord/gamma.psd.interm


/utmp/gamma_tr.4  is the directory of training data
/utmp/gamma_ts.4  is the directory of testing data
-t 4000    set the number of batches to 4000
-m 4000000 set the number of iterations (Recall that at each iteration, each machine randomly chooses a batch)
-c $2      set the regularizaiton parameter

Beginning from the first iteration,  it will save some statistics at every -t 4000 iteration (which is equivalent to full pass of data)
/utmp/jkrecord/gamma.psd.interm contains the temporary models 

/utmp/jkrecord/gamma.$1.tr.txt  contains the running time
/utmp/jkrecord/gamma.$1.ts.txt  contains the training and testing accuracy (Please ignore the objective value, which is computed based on ADMM objective, and I haven't modified it)


You can use the scripts in drawfig to parse the records, and draw a figure.

