#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include "linear.h"
#include "block.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#include <mpi.h>

void exit_with_help()
{
	printf(
	"Usage: parallel-predict [options] data_set_dir model_file\n"
	"Options: \n"
	"-p training_data_set_dir: evaluate primal value for each intermetidate result in the model_file \n"
    "-m number of predict: multiple number of prediction (default 1) \n"	
	);
	exit(1);
}

int main(int argc, char **argv)
{

	int rank;
	int size;	
    int num_predict = 1;
	char * test_set_dir, * train_set_dir;
	char  *intermediate_result_file;
	block_problem * bprob = new block_problem;
	MPI_Init(&argc, &argv);               // Initialize the MPI execution environment
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Determine current running process
	bprob->rank = rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes
	if (rank==0)
		printf ("Number of subsystems: %d \n", size);

	if (argc < 3)
		exit_with_help();
	
	train_set_dir = NULL;
	// parse options
	int i;
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'p':
				train_set_dir = argv[i];
				break;
			 
            case 'm':
				num_predict = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
    
	// determine filenames
	if(i+2!=argc)
		exit_with_help();

	test_set_dir = argv[i];
	intermediate_result_file = argv[i+1];


	// evaluate prediction accuracy
	
	struct model* model_;
	FILE * fp_model=NULL;

	bprob->read_metadata(test_set_dir);
		//printf ("Done in reading meta data \n");
	problem * prob;
	prob  = bprob->get_block(rank);	
	fp_model = fopen( intermediate_result_file, "rb");
	if (!fp_model)
		fprintf(stderr, "Cannot open file %s \n", intermediate_result_file);
	while (1)
	{
		// read the data	
		
		model_=load_model_intermediate(fp_model);
		if (!model_)
		{	
			break;
		}
		
		int correct=0;
		for (int i=0; i<prob->l; i++)
		{
            if (num_predict==1)
            {
                int predict_label = predict(model_, prob->x[i]);
                if (prob->y[i]== predict_label)
                    correct++;
            }
            else
            {
                int * predict_label_array = new int [num_predict];
                multi_predict(model_, prob->x[i], num_predict, predict_label_array);
                int hit = 0;
                for (int j=0; j< num_predict; j++)
                    if (prob->y[i]== predict_label_array[j])
                    {    
                        hit = 1;
                        break;
                    }
                if (hit == 1)
                    correct ++;
            }
				
		}
		int total_correct=0;
		MPI_Allreduce(&correct, &total_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (rank==0)
			printf("Accuracy = %g%% (%d/%d)\n",(double) total_correct/bprob->l*100,total_correct,bprob->l);
		
	}	
	fclose(fp_model);
	delete bprob;
	
	if (train_set_dir)
	{
		
		block_problem * bprob = new block_problem;
        // evaluate primal value if specified
		fp_model = fopen( intermediate_result_file, "rb");
		if (!fp_model)
			fprintf(stderr, "Cannot open file %s \n", intermediate_result_file);
		bprob->read_metadata(train_set_dir);
		 
		problem * prob;
		// read training data
		prob  = bprob->get_block(rank);
		//printf ("before, test: prob->y[0]:%d\n", prob->y[0]);
		// assign the +-1 label
		int start = bprob->start[ rank ];
		int *y = new int[bprob->l]; 	
		for(int j = start; j < start + prob->l; j++)
		// assign +-1 label
		{
			int lb = prob ->y[j-start];
			if(lb == bprob->label[0]) y[j] = 1;
			else if (lb == bprob->label[1]) y[j] = -1;
			else 
			{
				printf("id=%d, start = %d, l = %d datafmt = %d\n", rank , start, prob->l, bprob->datafmt);
				fprintf(stderr,"Do not support multiclass evaluation on primal value %d %d %d\n", lb, bprob->label[0], bprob->label[1]);
				exit(-1);
			}
		}	
		prob->y = y+start;
            
		// finish assigning label	
		while (1)
		{
			// read the data
			model_=load_model_intermediate(fp_model);
			if (!model_)
			{	
				break;
			}
            double send_train[3], recv_train[3];
			l2r_l2_svc_fun fun_obj(prob, model_->param.C, model_->param.C);
            send_train[0] = fun_obj.hinge_loss(model_->w, &send_train[1]);            //hinge_loss
			send_train[1] += 0.0; 
            send_train[2] = prob->l+0.0;
            //double sum_hloss = 0;
			MPI_Allreduce( send_train, recv_train, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                
            double w_norm = fun_obj.w_norm(model_->w);
			double pri_obj = w_norm + recv_train[0];
            double pri_accu = recv_train[1]/recv_train[2];
			if (rank==0)
				printf("primal obj %.8g train accuracy %.5g\n", pri_obj , pri_accu);

		}	
		fclose(fp_model);
		delete bprob;
	}
	MPI_Finalize(); 
	return 1;
}

  