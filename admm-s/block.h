#ifndef BLOCK_LINEAR
#define BLOCK_LINEAR

#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <time.h>
#include "linear.h"
 

using namespace std;

extern double start_t1;
extern time_t start_t;
//#extern double total_time;
void myfread(void *ptr, size_t size, size_t nmemb, FILE * stream);
int myuncompress(void *dest, size_t *destlen, const void *source, size_t sourcelen);

enum {BINARY, COMPRESSION}; // data format
class binaryfmt_problem
{
	public:
		int l, n;
		binaryfmt_problem(): l(0), n(0), buf(NULL), buflen(0), bias_idx(-1), bias(-1){}
		~binaryfmt_problem(){ if(buflen>0) free(buf);}

		void set_bias(int idx, double val, int datafmt = COMPRESSION);

		// load and transfer problem contents from "filename" to internal
		void load_problem(const char* filename, int datafmt);

		// return self as a (problem*) prob
		struct problem* get_problem();

	private:
		unsigned char* buf;
		size_t buflen, n_x_space, filelen;
		int bias_idx;
		double bias;
		struct feature_node* x_space;
		struct problem prob, retprob;

		void load_header(FILE *fp);
		void load_body(FILE *fp, int datafmt);
		void parse_binary();
};

class block_problem
{
	public:
		int nBlocks, n, l, nr_class;
		int datafmt;
		double bias;
		int rank;
		vector<string> binary_files;
		vector<int> start;
		// start of the instance index
		vector<int> subl;
		// l of the sub problem
		vector<int> label;
		binaryfmt_problem prob_;
		//binaryfmt_problem * biprob_list;
		block_problem(): n(0), l(0), datafmt(-1), bias(-1){}
		void set_bias(double b);

		// Read a directory generated by blockspliter
		void read_metadata(const char* dirname);

		// Get the (problem*) prob of the specified block
		struct problem* get_block(int id);		
		// Generate (block_problem) problem according to blocklist
		block_problem gen_sub_problem(const vector<int>& blocklist);
};


struct model* block_train(problem*  prob, const  parameter* param);
problem * read_problem(block_problem * prob, parameter* param);

double block_testing(struct model* model_, block_problem *bprob);
//double block_cross_validation(block_problem *bprob, const parameter *param, int nr_fold);
void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *total_correct);
const char *block_check_parameter(const block_problem *bprob, const parameter *param);

#endif
