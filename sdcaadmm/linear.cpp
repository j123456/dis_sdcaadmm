#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "linear.h"
//#include "tron.h"
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

//#if 0
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) 
{
	printf(fmt);
}
#endif

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	C = new double[l];
	I = new int[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else if (y[i] == -1)
			C[i] = Cn;
		else	
			{
				printf ("y should be +-1 \n ");
				exit(-1);
				
			}
		
	}
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	
	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

double l2r_l2_svc_fun::w_norm(double * w)
{
	double f=0;
	int w_size=get_nr_variable();
	for(int i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

double l2r_l2_svc_fun::hinge_loss(double *w, double * train_accu)
// L2-hinge loss
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
    * train_accu = 0;
	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		if (z[i]> 0)
            (*train_accu) += 1;
        double d = 1-z[i];
		if (d > 0)	
            f += C[i]*d*d;
          
	}


	return(f);
}

void l2r_l2_svc_fun::gradient_loss(double *w, double * grad_w)
// gradient of the square hinge loss
{
	int i;
	int *y=prob->y;
	int l=prob->l;	
    Xv(w, z);
	sizeI = 0;
    
	for (i=0;i<l;i++)
    {	
        z[i] = y[i]*z[i];        
		if (z[i] < 1)
		{
			z[sizeI] = 2* C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}        
    }
    
	subXTv(z, grad_w);
}


double l2r_l2_svc_fun::hinge_loss_2class(double *w, double * positive_accu, double * positive_total, double * negative_accu, double * negative_total)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
    * positive_accu = 0;
    * positive_total = 0;
    * negative_accu = 0;
    * negative_total = 0;
	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
        if (y[i]>0)
            (* positive_total) += 1;
        else
            (* negative_total) += 1;
		if (z[i]>=0)
        {
            if (y[i]>0)
                (*positive_accu) += 1;
            else    
                (*negative_accu) += 1;
        }
        double d = 1-z[i];
		if (d > 0)	
            f += C[i]*d*d;
          
	}


	return(f);
}
 

void l2r_l2_svc_fun::grad(double *w, double *g)
// You have to call fun before this functino, because it uses  z
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}


int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
// return X v which is a vector.
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;
     
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
         
	}
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

l2r_l2_proximity_fun::l2r_l2_proximity_fun(const problem *prob, double Cp, 
									double Cn, double * zz, double *uu)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	V = new double[n];
	D = new double[l];
	C = new double[l];
	I = new int[l];

	for (i=0; i<n; i++)
		V[i]= zz[i] - uu[i]; // v= z - u
	for (i=0; i<l; i++)
	{
		//V[i] = zz[i] - uu[i]; // v= z - u
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2r_l2_proximity_fun::~l2r_l2_proximity_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
	delete[] V;
}

double l2r_l2_proximity_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	
	Xv(w, z);
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<w_size;i++)
		f += (w[i] - V[i])*(w[i] - V[i]);
	f /= 2.0;

	return(f);
}

void l2r_l2_proximity_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i] - V[i];
}

int l2r_l2_proximity_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_proximity_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_proximity_fun::Xv(double *v, double *Xv)
// return X v which is a vector.
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_proximity_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_proximity_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

#define GETI(i) (prob->y[i])
// To support weights for instances, use GETI(i) (i)
 
int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}
 
#undef GETI
#define GETI(i) (y[i]+1)

// To support weights for instances, use GETI(i) (i)
 
// A coordinate descent algorithm for 
// Proximity, L1-loss and L2-loss SVM dual problems
//  Primal problem:
//  minimize [ C/rho*sum max(1-y_i w^T x_i,0) + 0.5||w - v||^2 ]  
//	Dual problem
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - t^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix, t_i=1- y_iv^Tx_i.
//
// In L1-SVM case:
// 		upper_bound_i = Cp/rho if y_i = 1
// 		upper_bound_i = Cn/rho if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = rho/(2*Cp)	if y_i = 1
// 		D_ii = rho/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn, rho, v, 
// eps is the stopping tolerance
//
// solution will be put in w


#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_proximity_l1l2_svc(
	const problem *prob, double *w, double *u,  
	double *alpha, double *z, double eps, 
	double Cp, double Cn, int solver_type,
	int max_iter, const struct parameter * param ){


	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	//int max_iter = 1000;
	int *index = new int[l];
	//double *alpha = new double[l];
	schar *y = new schar[l]; // This should be problematic if the number of class exceed 128
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{        
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}
	for(i=0; i<w_size; i++)
		w[i] = z[i] - u[i];
	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1; 
		}
		else
		{
			y[i] = -1;
		}
		QD[i] = diag[GETI(i)]; // Actually can be cached for further improvement

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			QD[i] += (xi->value)*(xi->value);
			w[xi->index-1]+= y[i]* (xi->value) * alpha[i];
			//assert(xi->index-1>=0 && xi->index-1<w_size);	 //Debug
			xi++;
		}
		index[i] = i;
	}

	
	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if (!param->inner_mute){
			if(iter % 10 == 0)
			info(".");
		}

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				if (!param->inner_mute){
					info("*");
				}
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}
 

	delete [] QD;
	// delete [] alpha;
	delete [] y;
	delete [] index;
}
 	
void sdca_admm(
	const problem *prob, double *w, double * y, double *y_old,
	double *alpha, double * q_aux, double * Zx, double * Zx_old,
	const struct parameter * param, int myrank, int numsmachine, double * tt)
{
	int i,j=0;
	int l = prob->l;
	int dim = prob->n;
	double rho_ad = param -> rho_ad;
	double gamma = param -> gamma;
	int total_block = param -> total_block;
	double eta_z = param -> eta_z;
	double eta_b = param -> eta_b;
	int g = numsmachine;
	

	double *tmp = new double[dim];

	double yy, uu, aux_u, By, By_old=0;
	double c = rho_ad*eta_z;

	if(myrank !=0 ){
		By = -y[myrank-1] + y[myrank];
		By_old = -y_old[myrank-1] + y_old[myrank];
	}
	else{
		By = y[myrank] - y[numsmachine-1];
		By_old = y_old[myrank] - y_old[numsmachine-1];
	}
//	printf("%f %f %f\n", w[0], Zx[0],By);

	//determine the index set 
	int block = rand()% total_block;
	int nums  = (int) floor( l / total_block);
	int begin = nums*block+1;
	int ends;

	if( block == total_block - 1){
		ends = l;
	}else{
		ends = nums*(block+1);
	}	
	double start1, start2, start3, start4;
	start1 = clock();//

	//compute: w_m - rho*(Zx + By) / (rho*z_eta)
	//maintain zx from outside 

	tt[0] += (clock()-start1)/CLOCKS_PER_SEC;//
	int idd = 0;
	double *alpha_old = new double[ends-begin];

	start2 = clock();//
	//loop the sampled batch to compute alpha_i
	for (j=begin; j<ends; j++)
	{
		if(prob->y[j] > 0)
		{
			yy = +1; 
		}
		else
		{
			yy = -1;
		}

		feature_node *xi = prob->x[j];
	        aux_u = alpha[j];
		alpha_old[idd] = alpha[j];

		while (xi->index != -1)
		{
			aux_u += (xi->value)*( w[xi->index-1] - rho_ad*(Zx[xi->index-1] + By) );

			//aux_u += (xi->value)*tmp[xi->index-1];
			//assert(xi->index-1>=0 && xi->index-1<w_size);	 //Debug
			xi++;
		}
		//proximal operation: smooth hinged loss
		uu = (c*aux_u*yy-1) / ( 1+c) ;			
		if( (-1 <= uu) && (uu <= 0) ){
			alpha[j] = (c*aux_u -yy)/(1+c);
		}else if( -1 > uu){
			alpha[j] = -yy;
		}else{
			alpha[j] = 0;
		}
		idd++;
	}
	tt[1] += (clock()-start2)/CLOCKS_PER_SEC;//

	start3 = clock();//

	//compute new Zx
	idd = 0;
	for (j=begin; j<ends; j++){
		feature_node *xi = prob->x[j];
		while (xi->index != -1)
		{
			Zx[xi->index-1] = Zx[xi->index-1] +  xi->value*(alpha[j]-alpha_old[idd]);
			//assert(xi->index-1>=0 && xi->index-1<w_size);	 //Debug
			xi++;
		}
		idd++;
	}
	tt[2] += (clock()-start3)/CLOCKS_PER_SEC;//

	start4 = clock();//

	//compute w, q_aux 
	for (i=0; i<g; i++)
		q_aux[i] = 0; 

	int N = l*numsmachine;
	double tmp2,tmp3 =0;
	for (i=0; i<dim; i++)
	{
	   if( (Zx[i]!=0) || (Zx_old[i] !=0) ){
		tmp2 = Zx[i] + By;
		//currently l is the number of samples in local machine
		w[i] = w[i] - gamma*rho_ad*( N*tmp2  - (N - N/total_block)*( Zx_old[i]+By_old )  ) ;
		tmp3 = (w[i] - rho_ad*tmp2)/(rho_ad*eta_b);

		if(myrank !=0 ){
			q_aux[myrank-1] -= tmp3; 
			q_aux[myrank] += tmp3;
		}else{
			q_aux[numsmachine-1] -= tmp3; 
			q_aux[myrank] += tmp3;
		}

	    }
	}
	tt[3] += (clock()-start4)/CLOCKS_PER_SEC;//

	delete [] tmp;
	delete [] alpha_old;
}


					
// A primal method to solve the proximity problem
// minimize [ C/rho*sum max(1-y_i w^T x_i,0)^2 + 0.5||w - v||^2 ], where v = z - u

void solve_proximity_l2_svc_primal( 
	const problem *prob, double *w, double * u,
	double *z, double eps, 
	double Cp, double Cn, const struct parameter * param)
{	 
	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i]==+1)
			pos++;
	
	neg = prob->l - pos;	
	function *fun_obj=NULL;
	fun_obj=new l2r_l2_proximity_fun(prob, Cp, Cn, z, u);	
	TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l, param->inner_max_iter, param->inner_mute, z);
	tron_obj.set_print_string(liblinear_print_string);
	tron_obj.tron(w);
	delete fun_obj;
}					
					
 
//
// Interface functions
//

void predict_values_preprocess(const struct model *model_, const struct feature_node *x, double *dec_values)
// given model and feature, predict the value for each class using w^T x.
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
			{	
                dec_values[i] += w[i* n +(idx-1)]*lx->value; // This could be a bug for multicalss with cross-validation. Need to fix it!                
            }
	}

	
}

int predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
// given model and feature, predict the value for each class using w^T x.
{
    predict_values_preprocess(model_, x, dec_values);
    int nr_class=model_->nr_class;
    if(nr_class==2)
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	else
	{
		int dec_max_idx = 0;
		for(int i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

void predict_multiple_values(const struct model *model_, const struct feature_node *x, double *dec_values, int num_predict, int * predict_label)
// given model and feature, predict the value for each class using w^T x.
{
    predict_values_preprocess(model_, x, dec_values);
    int nr_class=model_->nr_class;
    int * index_array = new int [nr_class];
    for (int i=0; i< nr_class; i++)
        index_array[i] = i;
    for (int i=0;i<nr_class-1;i++)
        for (int j=i+1;j<nr_class;j++)
        // sort by descending order
        {
            if(dec_values[index_array[i]] < dec_values[index_array[j]])
                swap(index_array[i], index_array[j]); // swap
        }
    for (int i=0; i< num_predict ; i++)
        predict_label[i] = model_->label[index_array[i]];

}

int predict(const model *model_, const feature_node *x)
// return the label of the feature given the model
{
	double *dec_values = Malloc(double, model_->nr_class);
	int label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

void multi_predict(const model *model_, const feature_node *x, int num_predict, int * predict_lable_array)
// return the top num_predict labels 
{
	double *dec_values = Malloc(double, model_->nr_class);
	predict_multiple_values( model_,  x,  dec_values, num_predict, predict_lable_array);
	free(dec_values);
}

int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		int label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;		
	}
	else
		return 0;
}


int save_model(const char *model_file_name, const struct model *model_ )
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter param = model_->param; 
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[j*w_size + i]); // This is different than LIBLINEAR
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

int save_model_intermediate(const char *model_file_name, const struct model *model_, FILE *fp )
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter param = model_->param; // note
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
 
	int nr_w;
	if(model_->nr_class==2 )
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);
	
	fprintf(fp, "C(single) %.16g\n", model_->param.C);
	
	//fprintf(fp, "Normalize %d\n", model_->param.normalize);
	
	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[j*w_size + i]);
		fprintf(fp, "\n");
	}
	
	return 1;
 
}


struct model *load_model_intermediate(FILE * fp)
{ 
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;
    model_->param.C = 1;
	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		if ( fscanf(fp,"%80s",cmd) == EOF)
            return NULL;
		if(strcmp(cmd,"solver_type")==0)
		{
			if ( fscanf(fp,"%80s",cmd) == EOF)
                return NULL;
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			if ( fscanf(fp,"%d",&nr_class) == EOF) 
                return NULL;
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			if ( fscanf(fp,"%d",&nr_feature) == EOF)
                return NULL;
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			if   ( fscanf(fp,"%lf",&bias) == EOF)
                return NULL;
			model_->bias=bias;
		}
		else if(strcmp(cmd,"C(single)")==0)
		{
			if ( fscanf(fp,"%lf",&(model_->param.C)) == EOF)
                return NULL;	
		}
 
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				if ( fscanf(fp,"%d",&model_->label[i]) == EOF)
                    return NULL;
		}
		else
		{
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			if ( fscanf(fp, "%lf ", &model_->w[j*w_size + i]) == EOF)
                return NULL;
		if ( fscanf(fp, "\n") == EOF)
            return NULL;
	}
 
	return model_;
}


int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC        
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		)
		return "checking parameter... unknown solver type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL) 
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

