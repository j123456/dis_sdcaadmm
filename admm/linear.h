#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include "tron.h"
#ifdef __cplusplus
extern "C" {
#endif
    
struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	int *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
	int *label;
    int nr_class;
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC,  L2R_L1LOSS_SVC_DUAL}; /* solver_type (0, 1 ,2 3, ..) */

// Sovler Table
static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", NULL
};    
    
struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double inner_eps;	/* inner stopping criteria */
	double C;
    double lambda;
	int nr_weight;
	int *weight_label;
	double* weight;
	int max_iter;
	int inner_max_iter;	
	char * intermediate_result;  
	double ABSTOL;
	double RELTOL;
	double rho;
	int relaxation;
	double relax_alpha;	
    double eta;	
	int inner_mute;
	int rank;
	int size;
	int primal;
	int root;
	int normalize;
    int SGD_decay;
    int PSGD_more_aver;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
int predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
int predict(const struct model *model_, const struct feature_node *x);
void multi_predict(const model *model_, const feature_node *x, int num_predict, int * predict_lable_array);
void predict_multiple_values(const struct model *model_, const struct feature_node *x, double *dec_values, int num_predict, int * predict_label);
void predict_values_preprocess(const struct model *model_, const struct feature_node *x, double *dec_values);
int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
int save_model_intermediate(const char *model_file_name, const struct model *model_, FILE *fp );

struct model *load_model_intermediate(FILE * fp);


int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));



void solve_proximity_l1l2_svc(
	const problem *prob, double *w, double *u,  
	double *alpha, double *z, double eps, 
	double Cp, double Cn, int solver_type,
	int max_iter, const struct parameter * param );
	
void solve_proximity_l2_svc_primal( 
	const problem *prob, double *w, double * u,
	double *z,  double eps, 
	double Cp, double Cn, const struct parameter * param);
	
 
 
class l2r_l2_svc_fun : public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double Cp, double Cn);
	~l2r_l2_svc_fun();

	double fun(double *w);
	double hinge_loss(double *w, double * train_accu);
    double hinge_loss_2class(double *w, double * positive_accu, double * positive_total, double * negative_accu, double * negative_total);
	double w_norm(double * w);
	void grad(double *w, double *g);
    void gradient_loss(double *w, double * grad_w);
	void Hv(double *s, double *Hs);
	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

class l2r_l2_proximity_fun : public function
{
public:
	l2r_l2_proximity_fun(const problem *prob, double Cp, double Cn, double *, double *);
	~l2r_l2_proximity_fun();

	double fun(double *w);
	double hinge_loss(double *w);
	void grad(double *w, double *g);	
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double * V;
	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};
#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

