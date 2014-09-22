// This program parses the command

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <mpi.h>
#include <cassert>
#include "linear.h"
#include "block.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
 
 
struct parameter param;
block_problem bprob;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;
  
void exit_with_help()
{
    if (param.rank==0) // only master node output information
        printf(
    "Usage: parallel-train [options] training_set_direcoty [model_file]\n"
    "options:\n"
    "-c cost : set the parameter C (default 1)\n"
    "-m max_iter : set the maximal number of ADMM iterations (default 50)\n"
    "-i intermediate_result_file : the file stores w for each outer iterations in the file \n"
    "-t number of mini-batches \n"
    "-p show primal objective:    1 -- yes , 0 -- no (default: 0)\n"
    "-g gamma (see paper) suggest to set to 1/ numbers of samples \n" 
    "-a rho   (see paper) \n"
    "-z \eta_z(see paper) \n" 
    "-b \eta_b(see paper) set to number of machines*features dimensions \n"
    );
    exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void do_cross_validation(const problem * prob);



int main(int argc, char **argv)
{
    char input_file_name[1024];
    char model_file_name[1024];
    const char *error_msg;
    /*
     * Some bookkeeping variables for MPI. The 'rank' of a process is its numeric id
     * in the process pool. For example, if we run a program via `mpirun -np 4 foo', then
     * the process ranks are 0 through 3. Here, N and size are the total number of processes 
     * running (in this example, 4).
    */
    start_t1 = clock(); //cpu time do not need declare type

    start_t2 = time(NULL);     //wall time
    MPI_Init(&argc, &argv);               // Initialize the MPI execution environment
    MPI_Comm_rank(MPI_COMM_WORLD, &param.rank); // Determine current running process
    MPI_Comm_size(MPI_COMM_WORLD, &param.size); // Total number of processes
    //double N = (double) size;             // Number of subsystems/slaves for ADMM
    if (param.rank==param.root)
        printf ("Number of subsystems: %d \n", param.size);
    
    parse_command_line(argc, argv, input_file_name, model_file_name);
    // Read the meta data
    bprob.read_metadata(input_file_name);
    bprob.set_bias(bias);
    error_msg = block_check_parameter(&bprob,&param);
    
    if(error_msg)
    {
        fprintf(stderr,"Error: %s\n",error_msg);
        exit(1);
    }
    
    if (param.rank==param.root)
    {    
        if (param.solver_type == L2R_L2LOSS_SVC)
            printf("ADMM + Primal trust region Newton's method for L2 loss SVM:\n");
        else if (param.solver_type == L2R_L2LOSS_SVC_DUAL)
            printf("ADMM + Dual coordinate descent for L2 loss SVM: \n");
        else if (param.solver_type ==  L2R_L1LOSS_SVC_DUAL)
            printf("ADMM + Dual coordinate descent for L1 loss SVM:\n");
        else
            printf("Not supported. \n"); 
    }
    
    srand(1);
    // Now read the local data 
    problem  * prob = read_problem(&bprob, &param);
    
    
    if(flag_cross_validation)
        do_cross_validation(prob);
    else
    {
        model_=block_train(prob, &param);   
        save_model(model_file_name, model_);  
        free_and_destroy_model(&model_);
    }
    destroy_param(&param);
    MPI_Finalize(); 
    return 0;
}

void do_cross_validation(const problem * prob)
{    
    int total_correct = 0;
    cross_validation(prob,&param,nr_fold,&total_correct);
    if (param.rank == param.root)
        printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/bprob.l);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i;
    // default values
    param.solver_type = L2R_L2LOSS_SVC_DUAL;
    param.C = 1.0;
    param.lambda = 0.0;
    param.eps = INF; // see setting below 
    param.max_iter = 50;
    param.inner_max_iter = 50;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;  
    param.intermediate_result =  NULL;
    flag_cross_validation = 0;
    bias = -1;
    param.RELTOL= 0.0001;
    param.ABSTOL=0.0001;
    param.rho=1.0;
    param.relaxation=0;
    param.relax_alpha=1.6;    
    param.inner_mute=1;     
    param.primal=0;
    param.normalize=0;
    param.root= 0;
    param.eta = 1E-3;
    param.SGD_decay = 0;
    param.PSGD_more_aver = 0;
    //new added 
    param.rho_ad = 0.01;
    param.gamma = 0.01;
    param.total_block = 100;
    param.eta_z = 1.2;
    param.eta_b = 1.2;
    // parse options
    for (i=1; i<argc; i++)
    {
        if (argv[i][0] != '-') break;
        if (++i>=argc)
            exit_with_help();
        switch( argv[i-1][1] )
        {
            case 't':
                param.total_block = atoi(argv[i]);
                break;
            case 'g':
                param.gamma = atof(argv[i]);
                break;
            case 'a':
                param.rho_ad = atof(argv[i]);
                break;
            case 'z':
                param.eta_z = atof(argv[i]);
                break;
            case 'b':
                param.eta_b = atof(argv[i]);
                break;


            case 's':
                param.solver_type = atoi(argv[i]);
                break;
            
            case 'c':
                param.C = atof(argv[i]);
                break;

            case 'e':
                param.eps = atof(argv[i]);
                break;

            case 'E':
                param.inner_eps = atof(argv[i]);
                break;
                
            case 'L':
                param.eta = atof(argv[i]);
                break;    

            case 'm':
                param.max_iter = atoi(argv[i]);
                break;
            
            case 'd':
                param.SGD_decay = atoi(argv[i]);
                break;                
            
            case 'n':
                param.normalize = atoi(argv[i]);
                break;
            
            case 'p':
                param.primal = atoi(argv[i]);
                break;    
            
            case 'F':
                param.PSGD_more_aver = atoi(argv[i]);
                break;
            
            case 'M':
                param.inner_max_iter = atoi(argv[i]);
                break;

            case 'B':
                bias = atof(argv[i]);
                break;
            case 'w':
                ++param.nr_weight;
                param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
                param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
                param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
                param.weight[param.nr_weight-1] = atof(argv[i]);
                break;

            case 'v':
                flag_cross_validation = 1;
                nr_fold = atoi(argv[i]);
                if(nr_fold < 2)
                {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    exit_with_help();
                }
                break;                
            case 'q':
                param.inner_mute=atoi(argv[i]);
                break;
            case 'R':
                param.RELTOL = atof(argv[i]);
                break;
                
            case 'A':
                param.ABSTOL = atof(argv[i]);
                break;                

            case 'r':
                param.rho = atof(argv[i]);
                break;                        
            
            case 'x':
                param.relaxation = atoi(argv[i]);
                break;        
                        
            case 'l':    
                param.relax_alpha= atof(argv[i]);
                break;
            
            case 'i':
                param.intermediate_result = argv[i];
                break;

            default:
                fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
                break;
        }
    }

    // determine filenames
    if(i>=argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if(i<argc-1)
        strcpy(model_file_name,argv[i+1]);
    else
    {
        size_t len = strlen(argv[i]);
        if(argv[i][len-1] == '/') 
            argv[i][len-1] = 0;
        char *p = strrchr(argv[i],'/');
        if(p==NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name,"%s.model",p);
    }   
    
    if(param.eps == INF)
    {
        if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
            param.eps = 0.01;
        else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL)
            param.eps = 0.001;

    }
}



