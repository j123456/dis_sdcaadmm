#include <math.h>

#include <algorithm>
#include <set>
#include <mpi.h>
#include <assert.h>
#include "zlib/zlib.h"
#include "block.h"
#include "linear.h"
#include <sys/types.h>
// used for open() file descriptor
#include <sys/stat.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>

time_t start_t2;
double start_t1;
const char *data_format_table[] = {
	"BINARY", "COMPRESSION", NULL
};

//#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Malloc(type,n) (type *)valloc((n)*sizeof(type))
#define Realloc(ptr, type, n) (type *)realloc((ptr), (n)*sizeof(type))
#define INF HUGE_VAL

//#define PRINT_POS_NEG //If you want to know the positive and negative accuracy

using namespace std;

void myfread(void *ptr, size_t size, size_t nmemb, FILE * stream) 
{
	size_t ret = fread(ptr, size, nmemb, stream);
	if(ret != nmemb) 
	{
		fprintf(stderr, "Read Error! Bye Bye %ld %ld\n", ret, size*nmemb);
		exit(-1);
	}
}

#define CHUNKSIZE 1073741824UL
int myuncompress(void *dest, size_t *destlen, const void *source, size_t sourcelen)
{
	int ret;
	z_stream strm;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	strm.avail_in = 0;
	strm.next_in = Z_NULL;
	ret = inflateInit(&strm);
	if (ret != Z_OK)
	{
		(void)inflateEnd(&strm);
		return ret;
	}

	unsigned char *in = (unsigned char *)source;
	unsigned char *out = (unsigned char *)dest;
	unsigned long bytesread = 0, byteswritten = 0;

	/* decompress until deflate stream ends or end of file */
	do {
		strm.avail_in = (uInt) min(CHUNKSIZE, sourcelen - bytesread);
		//finish all input
		if (strm.avail_in == 0)
			break;
		strm.next_in = in + bytesread;
		bytesread += strm.avail_in;

		/* run inflate() on input until output buffer not full */
		do {
			strm.avail_out = (uInt)CHUNKSIZE;
			strm.next_out = out + byteswritten;
			ret = inflate(&strm, Z_NO_FLUSH);
			assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
			switch (ret)
			{
				case Z_NEED_DICT:
					ret = Z_DATA_ERROR;     /* and fall through */
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					(void)inflateEnd(&strm);
					return ret;
			}
			byteswritten += CHUNKSIZE - strm.avail_out;
		} while (strm.avail_out == 0);

		/* done when inflate() says it's done */
	} while (ret != Z_STREAM_END);

	if(byteswritten != *destlen)
		fprintf(stderr,"Compressed file corrupted\n");
	*destlen = byteswritten;
	(void)inflateEnd(&strm);
	return 0;
}

// Body of class binaryfmt_problem
void binaryfmt_problem::set_bias(int idx, double val, int datafmt)
{
	if(bias >= 0) 
		fprintf(stderr, "Warning: the bias have been set to %lf\n.", bias);
	bias_idx = idx;
	bias = val;
}

void binaryfmt_problem::load_problem(const char* filename, int datafmt)
{
	int fd = open(filename, O_RDONLY);
	if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)==-1)
	// Used to advise OS that this file would not be read again.  
	{
		perror("posix_fadivse");
	}
	FILE *fp = fdopen(fd, "rb");  
	if (!fp)
	{
		printf("Cannot open the file %s \n", filename);
		exit(-1);
	}
	load_header(fp);
	load_body(fp, datafmt);
	parse_binary();
	l = prob.l;
	n = prob.n;
	close(fd);
	fclose(fp);
}

struct problem* binaryfmt_problem::get_problem()
{
	retprob = prob;
	if(bias >= 0 && prob.bias != bias)
	{
		struct feature_node node;
		prob.n = retprob.n = bias_idx;
		prob.bias = retprob.bias = bias;
		node.index = bias_idx;
		node.value = bias;

		for(int i=1;i<retprob.l;i++) 
			*(retprob.x[i]-2) = node; 
		x_space[n_x_space-2] = node;
	} 
	return &retprob;
}

void binaryfmt_problem::load_header(FILE *fp)
{
	myfread(&prob.l, sizeof(int), 1, fp);
	myfread(&prob.n, sizeof(int), 1, fp);
	myfread(&n_x_space, sizeof(unsigned long), 1, fp);
	myfread(&filelen, sizeof(unsigned long), 1, fp);
	prob.bias = -1;
	buflen = n_x_space * sizeof(struct feature_node) + prob.l * (sizeof(int)+sizeof(unsigned long));
}

void binaryfmt_problem::load_body(FILE *fp, int datafmt)
{	
	buf= Malloc(unsigned char, buflen);
	if (buf == NULL)
		fprintf(stderr,"Memory Error!\n");
	if(datafmt == BINARY)
	{
		if (buflen != filelen)
		{
			fprintf(stderr,"l = %d n_x_space = %ld buflen%ld filelen = %ld\n",prob.l, n_x_space, buflen, filelen);
		}
		myfread(buf, sizeof(unsigned char), buflen, fp);
	} 
	else if(datafmt == COMPRESSION)
	{
		unsigned char *compressedbuf;
		compressedbuf = Malloc(unsigned char, filelen);
		myfread(compressedbuf, sizeof(unsigned char), filelen, fp);
		int retcode = myuncompress(buf, &buflen, compressedbuf, filelen);
		if(retcode != Z_OK)
		{
			fprintf(stderr, "OK %d MEM %d BUF %d DATA %d g %d %p %ld\n", Z_OK, Z_MEM_ERROR, 
					Z_BUF_ERROR, Z_DATA_ERROR, retcode, buf, buflen);
		}
		free(compressedbuf);
	}
	if (madvise(buf, buflen,  MADV_SEQUENTIAL)!=0)
	{
		fprintf(stderr, "Error in using madvise\n");
	}
	
}

void binaryfmt_problem::parse_binary()
{
	unsigned long offset = 0;
	x_space = (struct feature_node*) (buf + offset); 
	offset += sizeof(struct feature_node) * n_x_space;

	prob.y = (int*) (buf + offset); 
	offset += sizeof(int) * prob.l;

	prob.x = (struct feature_node**) (buf + offset); 
	for(int i = 0; i < prob.l; i++) 
		prob.x[i] = x_space + (unsigned long)prob.x[i];
}

// Body of Class block_problem
void block_problem::set_bias(double b)
{
	bias = b;
	if(bias>=0) n+=1;
	prob_.set_bias(n, b, datafmt);
}

void block_problem::read_metadata(const char* dirname)
{
	char filename[1024], fmt[81];
	sprintf(filename,"%s/meta", dirname);
	FILE *meta = fopen(filename, "r");
	if(meta == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}
	if ( fscanf(meta, "%s", fmt) == EOF)
        exit(-1);  
	for(int i = 0; data_format_table[i]; i++)
		if (strcmp(data_format_table[i], fmt) == 0) 
			datafmt = i;
	if(datafmt == -1)
	{
		fprintf(stderr, "Unsupported binary data format\n");
		exit(-1);
	}
	if  ( fscanf(meta, "%d %d %d %d", &nBlocks, &l, &n, &nr_class) == EOF)
        exit(-1);
    
	label.resize(nr_class, 0);  // get the label
	for(int i = 0; i < nr_class; i++)
		if ( fscanf(meta, "%d", &label[i]) == EOF)
            exit(-1);

	binary_files.resize(nBlocks,"");
	start.resize(nBlocks,0);
	subl.resize(nBlocks,0);
	for(int i = 0; i < nBlocks; i++)
	{
		if ( fscanf(meta, "%d %d %s", &start[i], &subl[i], filename) == EOF)
            exit(-1);
		binary_files[i] = string(dirname) + "/" + string(filename); 
	}
	fclose(meta);

}


struct problem* block_problem::get_block(int id)
{
	if (id >= nBlocks)
	{
		fprintf(stderr,"Wrong Block Id %d; only %d blocks\n", id, nBlocks);
		exit(-1);
		return NULL;
	}
//    double startload_t = clock();
    time_t startload_t = time(NULL);
    prob_.load_problem(binary_files[id].c_str(), datafmt);
    double total_load = difftime(time(NULL), startload_t);
//    double total_load = ( clock() - startload_t) / CLOCKS_PER_SEC;
    printf ("Subsystem %d loads file %s in load time %.5g \n", id, binary_files[id].c_str(), total_load);
	return prob_.get_problem();
}
 


void normalize(struct problem * prob)
// use L2-norm for regularization
{
	int l=prob->l;
	feature_node **x=prob->x;

	for(int i=0;i<l;i++)
	{
		feature_node *s=x[i];
		double norm_sum=0.0;
		while(s->index!=-1)
		{
			norm_sum+= s->value * s->value;
			s++;
		}
		norm_sum = sqrt(norm_sum);
		if (norm_sum>1E-12){ 
            s = x[i];
            while(s->index!=-1)
            {
                s->value = s->value / norm_sum;
                s++;
            }
        }
        else 
            printf("See one all zero feature, some problem?\n");
	}
}
 
 
// Used only by CV at this moment
block_problem block_problem::gen_sub_problem(const vector<int>& blocklist)
{
	block_problem subbprob;
	subbprob.nBlocks = (int)blocklist.size();
	subbprob.n = n;
	subbprob.nr_class = nr_class;
	subbprob.bias = bias;
	subbprob.datafmt = datafmt;
	subbprob.label = label;

	subbprob.l = 0;
	for(int i = 0; i < subbprob.nBlocks; i++)
	{
		int bid = blocklist[i];
		subbprob.start.push_back(subbprob.l);
		subbprob.subl.push_back(subl[bid]);
		subbprob.binary_files.push_back(binary_files[bid]);
		subbprob.l += subl[bid];
	}

	return subbprob;
}

// class to handle intermeidate result
class intermediate_handler
{
	public:
		void push_item(void *ptr, size_t size) 
		{
			ptr_vec.push_back(ptr);
			size_vec.push_back(size);
		}
		void dump()
		// dump all things into disk
		{
			//FILE *fp = fopen(filename, "wb");
			if(fp == NULL)
			{
				fprintf(stderr,"can't open file \n");
				exit(1);
			}
			for(size_t i = 0; i < ptr_vec.size(); i++)
				fwrite(ptr_vec[i], 1, size_vec[i], fp);
			
			ptr_vec.clear();
			size_vec.clear();
			// clear the vector
			//fclose(fp);
		}
		void initialize(const char *filename)
		{
			fp= fopen(filename,"wb");
			if(fp == NULL)
			{
				fprintf(stderr,"can't open file %s\n",filename);
				exit(1);
			}
		}
		
		~intermediate_handler()
		{
			close();
		}
		intermediate_handler()
		{
			fp= NULL;
		}
		
		void close()
		{
			if (fp)
				fclose(fp);
		}

	private:
		FILE *fp;
		vector<void*> ptr_vec;
		vector<size_t> size_vec;
};

problem * read_problem( block_problem * bprob, parameter* param)
{
	
	int rank = param -> rank;
	problem * subprob  = bprob->get_block( rank  );
    subprob->n = bprob->n; // a subtlty here.
	subprob->nr_class = bprob->nr_class;
    subprob->label = new int [subprob->nr_class];
    for (int i=0; i< subprob->nr_class;i ++)
        subprob->label[i] = bprob->label[i];
	// <--normalized the data set if necessary
	if (param->normalize==1)
	{
		time_t normalize_time = time(NULL);
		//prob->normalize();
        normalize(subprob);
		printf("Subsystem %d normalize time %.5g \n", rank, difftime(time(NULL), normalize_time));
	}
	// -->
	fflush(stdout);
	return subprob;
}

void get_primal_val_accu(problem * prob, double  Cp, double  Cn, double * z, double * pri_obj, double * pri_accu) 
// Every node use z as the model and calcuate the primal objective based on its local data, use MPI_Alll reduce to get the sum.
{   
    l2r_l2_svc_fun fun_obj(prob, Cp, Cn);
    
    #ifndef PRINT_POS_NEG
        // only calculate primal value and training accuracy            
        double send_train[3], recv_train[3];         
        send_train[0] = fun_obj.hinge_loss(z, &send_train[1]); //hinge_loss
        send_train[1] += 0.0;  // send_train[1] is the accuracy
        send_train[2] = prob -> l+0.0;
        //double sum_hloss = 0;
        MPI_Allreduce( send_train, recv_train, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // Should be reduce
        //printf ("Finish MPI_Allreduce for primal objective\n"); //Debug
        double w_norm = fun_obj.w_norm(z);
        *pri_obj = w_norm + recv_train[0];
        *pri_accu = recv_train[1]/recv_train[2];
         
    #else        
        // calculate primal value and positive and negative accuracy
            
        double send_train[5], recv_train[5];
         
        send_train[0] = fun_obj.hinge_loss_2class(z, &send_train[1], &send_train[2], &send_train[3], &send_train[4]); //hinge_loss and pos/neg accuracy and total number
        MPI_Allreduce( send_train, recv_train, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //printf ("Finish MPI_Allreduce for primal objective\n"); //Debug
        double w_norm = fun_obj.w_norm(z);
        *pri_obj = w_norm + recv_train[0];
        *pri_pos_accu = recv_train[1]/(recv_train[2]+0.0);
        *pri_neg_accu = recv_train[3]/(recv_train[4]+0.0);
    #endif
}

void get_primal_gradient(problem * prob, double  Cp, double  Cn, double * w, double * tmp_grad_w, double * grad_w, double * grad_w_norm)
// Every node use z as the model and calcuate the gradient of the loss function based on its local data, use MPI_Alll reduce to get the sum and then get the gradient of the primal objective.
{   
    l2r_l2_svc_fun fun_obj(prob, Cp, Cn);    
    #ifndef PRINT_POS_NEG
        fun_obj.gradient_loss(w, tmp_grad_w); //hinge_loss
        MPI_Allreduce(tmp_grad_w, grad_w, prob -> n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        
        for (int j=0; j< prob -> n; j++)
        {    
            grad_w[j] += w[j]; //take into account of the regularization;
         
        }
        *grad_w_norm = fun_obj.w_norm( grad_w );       // This is actually twice of the norm.        
    #endif
}

void ADMM_L2R_solve (problem *prob, model * model_, double Cp, double Cn, int class_index=0)
// use ADMM frame to solve the problem in a distributed fashion. 
{
	srand (time(NULL));

	int l = prob->l;
	int n = prob->n;   
	parameter * param = &(model_-> param);
	double * w = (model_ -> w) + class_index* n ;    
	int solver_type = param->solver_type;
	int max_iter= param->max_iter;
	int inner_max_iter=param->inner_max_iter;
	double eps=param->eps;	
	double rho=param->rho;
	double ABSTOL = param->ABSTOL;
	double RELTOL = param->RELTOL;
	int relaxation= param->relaxation;	
	double relax_alpha = param-> relax_alpha;	
	int root =  param->root;
	int rank = param->rank;
	int total_block = param -> total_block;

	double rho_ad = param -> rho_ad;
	double eta_b  = param -> eta_b;
	double lambda = param -> C;
	double tmp = 0;

	// this alpha is to store the dual variables for each sub problem for dual method	
	
	double *alpha = NULL;	
        alpha = new double[l];
        for (int i=0; i<l; i++)
        	alpha[i]= 0; //initaillize

	double *dnorm = NULL;	
        dnorm = new double[l];

        for (int j=0; j<l; j++){
		feature_node *xi = prob->x[j];
	        tmp = 0;
		while (xi->index != -1)
		{
			tmp += (xi->value)*(xi->value);
			xi++;
		}
		dnorm[j]= tmp; //initaillize
	}

	double * z= new double[n];
	double * aux_z= new double[n];
	double * z_old= new double[n];

	int iter = 0;	
	double N_subsystem = (double) param-> size; 
	int numsmachine = param-> size;
	
 	// initialization
	for (int j=0; j< n; j++)
	{
			w[j]=0;
			z[j]=0; 
			aux_z[j]=0; 
			z_old[j]=0; 
	}

		
        double startmisc_t, startrun_t;
	double total_time1, total_time2=0.0;    
        double total_misc=0.0, total_run=0.0;
/*
	double * tt= new double[4];
	for (int j=0; j<4; j++){
		tt[j] = 0;
	}
*/
	// <-- Initialize intermediate result write
	FILE *fp = NULL;
	if (param->intermediate_result != NULL)   //&& rank==0)
	{	
		fp = fopen(param->intermediate_result,"wb");
		if(fp==NULL) 
		{
			printf("Cannot open the intermeidate file %s \n", param->intermediate_result);
			exit(-1);
		};
		//intermediate_result.initialize(param->intermediate_result);
	}
	// -->
//	printf("hello \n");
	// <-- Start of processing data using ADMM + Coordinate descent / Trust Region Newton
	while (iter < max_iter)
	{
//		printf("Iter %d, Max_iter %d\n", iter, max_iter);
		startrun_t = clock();
		iter++;

		psdca(prob, w, aux_z, dnorm, alpha, param);

		total_run += (clock()-startrun_t)/CLOCKS_PER_SEC;

		startmisc_t = clock();

		#ifdef DEF_MPIREDUCE
		MPI_Allreduce(aux_z, z, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		#else
		MPI_Reduce(aux_z, z, n, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD); 
        	#endif	
		total_misc += (clock()-startmisc_t)/CLOCKS_PER_SEC;

		startrun_t = clock();
		for (int j=0; j<n; j++)
			z[j] += z_old[j]; 
		total_run += (clock()-startrun_t)/CLOCKS_PER_SEC;


		if (!param->inner_mute)
		{
			printf(" \n");
			fflush(stdout);
		}

		if( (iter == max_iter) || ( (iter-1)%total_block == 0) ){
		      	if (rank==root){
				total_time1 = (clock()-start_t1)/CLOCKS_PER_SEC;
				total_time2 = difftime(time(NULL), start_t2);		
//	   	    		#ifndef PRINT_POS_NEG
        	        	printf("iter %d totaltime %.5g cputime %f  runtime %f coortime %f\n", iter, total_time2, total_time1, total_run, total_misc);
//       	        	printf("t1 %f t2 %f t3 %f t4 %f\n", tt[0], tt[1], tt[2], tt[3]);
//	     	    		#else
//	   	        	printf("iter %d time %f \n", iter, total_time);
//        	    		#endif
			}

			//MPI broadcast
			startmisc_t = clock();
			MPI_Bcast(z, n, MPI_DOUBLE, root, MPI_COMM_WORLD);
			total_misc += (clock()-startmisc_t)/CLOCKS_PER_SEC;

			if (param->intermediate_result) // The intermediate file will be written to every node. 
			{ 
				// MPI communication: averages the w			
				model_ ->w = z;
				save_model_intermediate(param->intermediate_result, model_, fp);
	            		// this should not be used for multiclass classification though
				model_ ->w = w;
			}

			fflush(stdout);
		}
		for (int j=0; j<n; j++)
			z_old[j] = z[j]; 
	} //end while
        if (param->intermediate_result)
	{	
		if (fclose(fp)!=0)
		{
			printf("Fail to close the file %s \n", param-> intermediate_result);
			exit(-1);
		}
	}
/*
    for (int i=0; i<n; i++)
        w[i] = z[i];
*/	
	if (alpha)
		delete [] alpha;
	if (dnorm)
		delete [] dnorm;
	
	delete [] z;
	delete [] z_old;
	delete [] aux_z;
}
 
  
void block_train_one(problem * prob, model * model_, double Cp, double Cn, int class_index=0)
{
	switch (model_->param.solver_type)
	{
		case L2R_L1LOSS_SVC_DUAL: // use ADMM + DCD
			ADMM_L2R_solve( prob, model_, Cp, Cn, class_index);
			break;
		case L2R_L2LOSS_SVC_DUAL: 
			ADMM_L2R_solve( prob, model_, Cp, Cn, class_index);
			break;       
		case L2R_L2LOSS_SVC: // use ADMM + trust Newton
			ADMM_L2R_solve( prob, model_, Cp, Cn, class_index);
			break;
            
		default:
			fprintf(stderr,"Not support for block version!\n");
	}
}


// This is the interface function, given the feature and labels (problem), and the parameters, it will return the model. Note that the labels are not processed yet, i.e., they can be other than +1 -1, and multiclass. 
struct model* block_train(problem*  prob, const  parameter* param)
{
	//prob->rank = param->rank;
	int n = prob->n;
	int l = prob->l;
	model *model_ = Malloc(model,1);
	
	if( prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias =  prob->bias;

    int nr_class= prob->nr_class;
    int * label = prob->label;
    
    model_->nr_class = nr_class;
	model_->label = Malloc(int, nr_class);
    for(int i=0;i<nr_class;i++)
		model_->label[i] = label[i];
    
	// <-- calculate weighted C
	double *weighted_C = Malloc(double, nr_class);

	for(int i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(int i=0;i<param->nr_weight;i++)
	{
		int j;
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}
	// -->
	// <--constructing the subproblem
	feature_node **x = Malloc(feature_node *,l);
    for(int i=0;i<l;i++)
		x[i] = prob->x[i];
	int k;
	problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.x = Malloc(feature_node *,sub_prob.l);
	sub_prob.y = Malloc(int,sub_prob.l);
    sub_prob.label= label;
    sub_prob.nr_class = nr_class;  
	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];
	// -->
    // <-- start using solvers
	 	
    if(nr_class == 2)
    {
        model_->w=Malloc(double, n);
        for(int j = 0; j < l; j++)
        // assign +1 -1 label
        {
            int lb = prob ->y[j];
            if(lb ==  label[0]) sub_prob.y[j] = 1;
            else if (lb ==  label[1]) sub_prob.y[j] = -1;
            else 
            {             
                fprintf(stderr,"The label is wrong %d %d %d\n", lb,  label[0],  label[1]);
                exit(-1);
            }
        }	
        // -->     
         
        block_train_one( &sub_prob,  model_, weighted_C[0], weighted_C[1], 0);
    } 
    else 
    {
        model_->w=Malloc(double, n*nr_class);            
        for(int i=0;i<nr_class;i++)
        {
            for (int j=0; j<l; j++)
            {
                int lb= prob-> y[j];
                if (lb == label[i]) sub_prob.y[j]=  1;
                else 
                {                        
                        sub_prob.y[j]= -1;                       
                }
            }
            // finish assign labels
            block_train_one(&sub_prob, model_, weighted_C[i], param->C, i);
        }           
    }
 
	// -->
    
    free(x);
	free(sub_prob.x);
	free(sub_prob.y);
    
	free(weighted_C);	
	return model_;
}

struct model* block_test(block_problem* bprob, const  parameter* param)
{
	bprob->rank = param->rank;
	int n = bprob->n;
	model *model_ = Malloc(model,1);
	
	if(bprob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = bprob->bias;

	int nr_class = bprob->nr_class;
	vector<int> &label = bprob->label;

	model_->nr_class = nr_class;
	model_->label = Malloc(int, nr_class);
	for(int i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);

	for(int i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(int i=0;i<param->nr_weight;i++)
	{
		int j;
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}


	
    if(nr_class == 2)
    {
        model_->w=Malloc(double, n);			
    } 
    else 
    {
        fprintf(stderr, "Use -s 4 for Multiclass-SVM\n");
        exit(-1);
    }
	

	free(weighted_C);
	return model_;
}

double block_testing(struct model *model_, block_problem *bprob)
{
	int correct = 0;
	for(int i=0; i<bprob->nBlocks; i++)
	{
		struct problem *subprob = bprob->get_block(i);
		for(int j=0;j<subprob->l;j++)
			if (predict(model_, subprob->x[j]) == subprob->y[j])
				correct++;
	}
	return (double)correct;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *total_correct)
{
    * total_correct = 0;    
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
    // <--randomize the data
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
    // -->
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(int,subprob.l);
        subprob.label= prob->label;
        subprob.nr_class = prob->nr_class;

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = block_train(&subprob,param);
        int correct=0;
		for (j=begin;j<end;j++)
		{
			int predict_label = predict(submodel, prob->x[perm[j]]);
			if (prob->y[perm[j]]== predict_label)
				correct++;	
		}
        int recv_sum;
		MPI_Allreduce(&correct, &recv_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        *total_correct += recv_sum;
        free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);


}

const char *block_check_parameter(const block_problem *bprob, const parameter *param)
{
	const char *error_msg = check_parameter(NULL,param);
	if(error_msg)
		return error_msg;    
	return NULL;
}

