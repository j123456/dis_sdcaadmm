#! /usr/bin/python
#####################
# This script will split the data and spread them in the set directory to each node specified by host files
# The script makes the following assumption:
# (0) This script runs on a master node
# (1) The split file has the same absolute directory across all machines
# (2) Password-less ssh from master to slaves

####################
import sys
import subprocess
import os

#node_list=[node1,node4,node5,node6,node7,node8, node9]
if len(sys.argv)!= 3:
    print "Usage %s host_file data_file" % (sys.argv[0])
    sys.exit(-1)
# example %s ~/.mpi_host ~/VLSC/virus/train.dat.scale.8(/) ~/VLSC/virus/train.dat.scale.8(/)
data_file = os.path.abspath( sys.argv[2] )
host_filename= sys.argv[1]
lines = open(host_filename,'r').readlines()
node_list=[]
package_path = os.getcwd() #os.path.dirname (sys.argv[0])
for line in lines:
    if line!='\n':
        if (not '#' in line) and ('node' in line):
            node_list.append(line.split()[0])

num_of_nodes= len(node_list)
node_list.pop(0)
# remove the first element which is the master node
#print node_list        

# Split the data
split_data_dir = "%s.%d" % (data_file, num_of_nodes)
# Remove the directory if exisits
if os.path.exists(split_data_dir):
    cmd = 'rm -r %s' % (split_data_dir)
    subprocess.Popen(cmd, shell = True).communicate()
    
blocksplit_path = package_path + '/blocksplit'
cmd = '%s -m %d %s' % ( blocksplit_path,  num_of_nodes, data_file)
print "Spliting file..."
print "Executing :" + cmd
subprocess.Popen(cmd, shell=True).communicate()
#  Now spread the files

source_dir = split_data_dir
target_dir = split_data_dir
line = open( source_dir+'/meta', 'r' ).readlines()[1]
n = int(line.split()[0])

# make target directory for remote nodes if necessary
print '\nSpreading...'
print 'Making target directory for remote nodes if necessary'
for i in range(n-1):    
    cmd = 'ssh %s -P mkdir -p %s/data' % (node_list[i], target_dir)
    print cmd
    pid = subprocess.Popen(cmd, shell = True).communicate()

# Open multiple processes in the background    
process_id = [] 
for i in range(n-1):
    bin_filename = '/data/'+str(n)+'_'+str(i+2)+'.bin'
    cmd = "scp %s %s:%s" % (source_dir+ '/meta', node_list[i], target_dir)
    print cmd
    subprocess.Popen(cmd, shell= True)    
    cmd = "scp %s %s:%s" % (source_dir+bin_filename, node_list[i], target_dir + bin_filename )
    print cmd
    process_id.append( subprocess.Popen(cmd, shell=True) )

# Poll them     
while (1):    
    all_done = 1;
    for i in range(n-1):
        process_id[i].poll()
        if ( process_id[i].returncode ==None ):
            all_done = 0;
            break;
    if all_done == 1:
        break;

print
print "Done in spliting and spreading files."
print "The data file is in the directory " + split_data_dir
