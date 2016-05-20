#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
// use clock_t
#include <time.h>

// use DBL_MAX
#include <float.h>
// use armadillo library
// #define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include "svm.h"
using namespace arma;

int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;

// global variables
double *Global_Alpha;
// double *Check_Alpha;
double Global_Rho=0;
// Qfloat **Global_Q;
double tolerance_M=0.08;
double variable_C = 0.04;
double tolerance_C = 0.01;
double tolerance_em = 0.01;

// TODO
int iterations_check = 0;
int iterations_libsvm = 0;
int iterations_replacement = 0;
double libsvm_svm_train_time = 0;
double replacement_svm_train_time = 0;
double time_comsuming_solve = 0;
double time_comsuming_solve_c = 0;



#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

// redefine the dot function for caculating f(x_i) with poly kernel
static inline double Kernel_dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
// test: bean
// printf("the kernel_type is %d\n", param.kernel_type);

	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			{
// printf("get in POLY\n");
				return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
			}
		case RBF:
		{
// printf("get in RBF\n");
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);
};

void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
	}

	// optimization step

	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;
	
	while(iter < max_iter)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	if(iter >= max_iter)
	{
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	// sum of iter
	iterations_check = iter;

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps || Gmin_idx == -1)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver::do_shrinking()    // for jiasu training
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free; // mean
	else
		r = (ub+lb)/2;	// middle, ub, lb

	return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU: public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))	
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps || Gmin_idx == -1)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;
	
	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	// return *len* values in row i, but this is just about subprob, not all
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i); 
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(this->*kernel_function)(i,j);
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			for(j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

static void solve_c_svc_origin(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
printf("		>>>>>>>>>> get into solve_c_svc_origin\n");
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;
// printf("in solve_c_svc\n");
	for(i=0;i<l;i++)
	{
// printf("the index is: %d and the y[%d] is: %lf\n", i, i,prob->y[i]);	
		// alpha[i] = 0;
		minus_ones[i] = -1;
		// used perm when call svm_train function, no need to use perm again here
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
		// alpha[i] *= y[i];
	}

	double check_KKT_round = 0;
	for (int i = 0; i < l; ++i)
	{
		check_KKT_round+=prob->y[i]*alpha[i];
	}
printf("in solve_c_svc_origin after a round, check_KKT_round = %lf\n", check_KKT_round);

	Solver s;

clock_t start_train_solve = clock(), end_train_solve;


	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

end_train_solve = clock();
time_comsuming_solve += (double)(end_train_solve-start_train_solve)/CLOCKS_PER_SEC;
printf("elasped time for Solve() is: %lfs \n", (double)(end_train_solve-start_train_solve)/CLOCKS_PER_SEC);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));
	// set Global_Alpha
	// Global_Alpha = new double[l];
	// Global_Alpha = alpha;

	// alpha multiples y[i]

	for(i=0;i<l;i++)
	{
		alpha[i] *= y[i];
	}


	delete[] minus_ones;
	delete[] y;
printf("		<<<<<<<<< get out of solve_c_svc_origin\n");
}

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;
// printf("in solve_c_svc\n");
	for(i=0;i<l;i++)
	{
// printf("the index is: %d and the y[%d] is: %lf\n", i, i,prob->y[i]);	
		alpha[i] = 0;
		minus_ones[i] = -1;
		// used perm when call svm_train function, no need to use perm again here
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);


	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));
	// set Global_Alpha
	// Global_Alpha = new double[l];
	// Global_Alpha = alpha;

	// alpha multiples y[i]

	for(i=0;i<l;i++)
	{
		alpha[i] *= y[i];
	}


	delete[] minus_ones;
	delete[] y;
}


static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;
};

static decision_function svm_train_one_origin(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double *whole_Alpha)
{
printf("	>>>>>>>>>>>>> get into svm_train_one_origin\n");
	double *alpha = Malloc(double,prob->l);
	alpha = whole_Alpha;
	Solver::SolutionInfo si;

double sum_y_a = 0;
for (int i = 0; i < prob->l; ++i)
{
	sum_y_a += prob->y[i]*whole_Alpha[i];
}
printf("in svm_train_one_origin sum{alpha_i * y_i} = %lf\n", sum_y_a);

	switch(param->svm_type)
	{
		case C_SVC:
		{
clock_t start_train_solve_c = clock(), end_train_solve_c;

			solve_c_svc_origin(prob,param,alpha,&si,Cp,Cn);

end_train_solve_c = clock();	
time_comsuming_solve_c += (double)(end_train_solve_c-start_train_solve_c)/CLOCKS_PER_SEC;
printf("elasped time for solve_c_svc_origin() is: %lfs\n", (double)(end_train_solve_c-start_train_solve_c)/CLOCKS_PER_SEC);
		}
			
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// set rho to a global variable
	// Global_Rho = si.rho;

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;

	f.rho = si.rho;
printf("	<<<<<<<<<<<<<< get out of svm_train_one_origin\n");
	return f;
}

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// set rho to a global variable
	Global_Rho = si.rho;

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;

	// set the global alpha, XXX
	// Global_Alpha = alpha;
	f.rho = si.rho;
	return f;
}


// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter;
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam);
			for(j=begin;j<end;j++)
			{
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]]));
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
// test
// printf("in svm_group_classes\n");
// for(i=0;i<l;i++)
// {
// printf("the index is: %d and the perm[%d] is: %d, y[%d] is: %lf, y[perm[%d]]~y[%d] is: %lf\n", i, i, perm[i], i, prob->y[i], i, perm[i], prob->y[perm[i]]);	
// }
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
printf(">>>>>>>>>> get into svm_train!\n");
	// prob->x or prob->y, are permutated one fold after another
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
// printf("22222222222, second calling svm_group_classes\n");
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		if(nr_class == 1) 
			info("WARNING: training data in only one class. See README for details.\n");
		
		// perm is permutated, starts with all positive instances, and then with all negtive instances
		svm_node **x = Malloc(svm_node *,l);
		int i;
		for(i=0;i<l;i++)
		{
			// due to perm, indices start with all positive instances, followed by all negtive instances
			x[i] = prob->x[perm[i]];
			// printf("i = %d, perm[i]: perm[%d]=%d and y[%d] = %lf y[perm[%d]]=%lf\n", i, i, perm[i], i, prob->y[i], i, prob->y[perm[i]]);
		}

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		// out loop i, inner loop j, and j=i+1
		// so all the times of loop is 1+2+3+...+(k-1) = k*(k-1)/2

		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for(i=0;i<nr_class;i++)		// nr_class = 2 in this case.
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				// si starts with positive instance, +1
				// sj starts with negtive instance, -1

				// printf("test set start at:%d \n", si);
				// printf("test set end with:%d \n", sj);
				// printf("the subprob->l = %d\n", l);
				int ci = count[i], cj = count[j];
				// printf("count of si:%d \n", ci);
				// printf("count of sj:%d \n", cj);
				// printf("si+sj=%d \n", ci+cj);
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);
// printf("in svm_train\n");				
// for (int ww = 0; ww < sub_prob.l; ++ww)
// {
// printf("the index is: %d and the y[%d] is: %lf\n", ww, ww, sub_prob.y[ww]);	
// }

				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				
				// set Global_Alpha, nr_class = 2, this loop just work once.
				// bool check = true;

				Global_Alpha = new double[l];
				for (int i = 0; i < l; ++i)
				{
					Global_Alpha[perm[i]]=f[p].alpha[i]*sub_prob.y[i];
					// Check_Alpha
					// if(Global_Alpha[perm[i]]<0)
					// 	check = false;

// printf("in svm_train, i = %d, Global_Alpha[perm[i]]~Global_Alpha[perm[%d]]=%lf, y[%d] = %lf, perm[%d] = %d,  y[perm[%d]] = %lf f.alpha[%d] = %lf\n", i, i, Global_Alpha[perm[i]], i, prob->y[i], i, perm[i], i, prob->y[perm[i]], i, f[p].alpha[i]);
				}
// printf("check the alpha[i] is negtive or not: %d\n", check);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}
// printf("ppppppppppp = %d \n", p);

		// after for, p = 1, when nr_class = 2



		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
printf("<<<<<<<<<< get out of svm_train!\n");
	return model;
}


// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;

		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// perm here is permutated to all 1 before all -1
		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)	// nr_class = 2 when C_SVC
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	// perm here is permutated by one fold after anoter, 1 before -1

	// local variable for storing alpha
	double *whole_Alpha = new double[l];

	for (int i = 0; i < l; ++i)
	{
		whole_Alpha[i] = 0;
	}

	int begin0 = fold_start[0];
	int end0 = fold_start[0+1];

	double time_comsuming = 0;
	double time_comsuming_train = 0;

	int first_round_l = 0;
	// first round of cross validation
	{
		// check index of begin0 and end0
		printf("first round--> begin0:%d\n",begin0);
		printf("first round--> end0: %d\n",end0);

		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end0-begin0);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
		
		first_round_l = subprob.l;
		k=0;
		for(j=0;j<begin0;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end0;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}

		clock_t start_train = clock(), end_train;
		struct svm_model *submodel = svm_train(&subprob,param);		// subprob is just traing data, not contained test data which indices from begin0 to end0.
		end_train = clock();
		time_comsuming_train+=(double)(end_train-start_train)/CLOCKS_PER_SEC;
		printf("elasped time for svm_train() is: %lfs, current fold is %d\n", (double)(end_train-start_train)/CLOCKS_PER_SEC, 1);

		// Global_Alpha is just stored alpha of subprob, not all data
		// so give all alpha to whole_Alpha
		// the alphas between begin0 and end0 are initialized to zero.
		// get Global_Alpha in function svm_train_one
double check_KKT = 0;
printf("at first, check_KKT = %lf\n", check_KKT);
		for (int i = 0; i < begin0; ++i)
		{
			whole_Alpha[perm[i]] = Global_Alpha[i];
			check_KKT+=subprob.y[i]*whole_Alpha[perm[i]];
		}

		for (int i = begin0; i < subprob.l; ++i)
		{
			whole_Alpha[perm[end0+i]] = Global_Alpha[i];
			check_KKT+=subprob.y[i]*whole_Alpha[perm[end0+i]];
		}

printf("at last, check_KKT = %lf\n", check_KKT);


		// predict the test set, and calculate accuracy
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin0;j<end0;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);
		}
		else
			for(j=begin0;j<end0;j++)
			{
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
			}
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}





	
	// TODO
	// double* fixed_fisrt_alpha = new double[first_round_l];
	// for (int i = 0; i < first_round_l; ++i)
	// {
	// 	fixed_fisrt_alpha[i]=Global_Alpha[i];
	// }
	// set parameter C to variable_C
	variable_C = param->C;

	// set first_rho
	double first_rho = Global_Rho;

	// Kernel* kernel = new Kernel(l, kx, param);
printf("================= end of first round =====================\n");
	// Q_ij, memory allocation
	Qfloat **all_Q = new Qfloat*[l];
	// schar *ys = new schar[l];
	// Qfloat **all_Q1 = new Qfloat*[l];
	for(int i=0;i<l;i++)
	{
		all_Q[i] = new Qfloat[l];
	}

	get_Qij(prob, param, all_Q);

	int cur_fold = 1;
	// the rest rounds
	for(i=1;i<nr_fold;i++) // nr_fold
	{
cur_fold = i;
printf("-=-=-=-=-=-=-=-> this is #%d round\n", i);
		
		// Set A, the testing set of previous round is just A in this round
		int begin_A = begin0;
		int end_A = end0;
printf("the set A starts at: %d, ends at %d in this round\n", begin_A, end_A);

		// record the valid indices in A
		int *valid_A = new int[end_A-begin_A];
		int *index_A = new int[end_A-begin_A];
		int count_A = end_A-begin_A;

		for(int a=0;a<end_A-begin_A;a++)
		{
			// if the index is valid, set 1, otherwise set 0;
			valid_A[a]=1; 
			index_A[a]=begin_A+a;
		}
		// Set R, remove testing subset in current round
		int begin = fold_start[i];
		int end = fold_start[i+1];

printf("the set R starts at: %d, ends at %d in this round\n", begin, end);

		for (int m = begin_A; m < end_A; ++m)
		{
			whole_Alpha[perm[m]] = 0;
		}

printf("first_round_l = %d end_A-begin_A = %d sum = %d \n", first_round_l, end_A-begin_A, end_A-begin_A+first_round_l);

		// for (int n = 0; n < first_round_l; ++n)
		// {
		// 	whole_Alpha[perm[end_A+n]] = fixed_fisrt_alpha[n];
		// }

		// reset Global_Rho
		Global_Rho = first_rho;

		// int *R = new int[end-begin];
		int *valid_R = new int[end-begin];
		int *index_R = new int[end-begin];
		// initialize 0
		for (int i = 0; i < end-begin; ++i)
		{
			valid_R[i] = 1;
			index_R[i] = begin+i;
		}
		int count_R = end-begin;
		// int update_count_R = 0;
		// int fixed_count_R = end-begin;

		// for(int b=begin;b<end;b++)
		// {
		// 	// R[b-begin]=b;
		// 	if(whole_Alpha[perm[b]]>0)
		// 	{
		// 		valid_R[b-begin]=1;
		// 		count_R++;
		// 	}
		// }

		int *valid_M = new int[l];
		int count_M = 0;

		int *valid_O = new int[l];
		int count_O = 0;

		int *valid_I = new int[l];
		int count_I = 0;
		
		// initialize arrays
		for (int i = 0; i < l; ++i)
		{
			valid_M[i]=0;
			valid_O[i]=0;
			valid_I[i]=0;
		}

		int count_ignored_R = 0;
		int count_ignored_A = 0;


		for(int fi=0;fi<l;fi++)
		{
			if(fi>=begin && fi<end)
			{
				count_ignored_R++;
				continue;
			}

			if(fi>=begin_A && fi<end_A)
			{
				count_ignored_A++;
				continue;
			}
			
			// calculate f(x_i), use all data
			double fxi = get_Fxi(perm[fi], prob, param, whole_Alpha, Global_Rho);

			double condition = prob->y[perm[fi]]*fxi;
			// origin perm

			// Conditional jump or move depends on uninitialised value(s)
			if( condition >= 1-tolerance_M && condition <= 1+tolerance_M )
			{// M
				// if i belongs to A, and y_i*f(x_i)=1, remove i from A, add it to M
				// if the instance of index fi has been removed at previous round, just ignore it.
				valid_M[fi] = 1;
				count_M++;
			}
			else if(condition > 1+tolerance_M)
			{// O
				// if i belongs to A, and y_i*f(x_i)>1, remove i from A, add it to O
				// if the instance of index fi has been removed at previous round, just ignore it.
				valid_O[fi] = 1;
				count_O++;
			}
			else
			{// I
				valid_I[fi] = 1;
				count_I++;
			}
		}

printf("count_ignored_R+count_ignored_A+count_M+count_O+count_I = %d\n", count_ignored_R+count_ignored_A+count_M+count_O+count_I);

	

		int *index_M = new int[count_M];
		int *index_O = new int[count_O];
		int *index_I = new int[count_I];
		// int *index_A = new int[count_A];
		// int *index_R = new int[count_R];

		// Set M, record the valid indexes
		int out_m = 0;
		for(int i=0;i<l;i++)
		{
			if(i>=begin && i<end)
			{
				continue;
			}

			if(i>=begin_A && i<end_A)
			{
				continue;
			}

			if(valid_M[i]>0)
			{
				// record the valid index
				index_M[out_m++]=i;
			}
		}
printf("check if count_M==out_m: %s\n", (count_M==out_m)? "true": "false");

		// Set O, record the valid indexes
		int out_o = 0;
		for(int i=0;i<l;i++)
		{
			if(i>=begin && i<end)
			{
				continue;
			}

			if(i>=begin_A && i<end_A)
			{
				continue;
			}

			if(valid_O[i]>0)
			{
				// record the valid index
				index_O[out_o++]=i;
			}
		}
printf("check if count_O==out_o: %s\n", (count_O==out_o)? "true": "false");

		// Set I, record the valid indexes
		int out_i = 0;
		for(int i=0;i<l;i++)
		{
			if(i>=begin && i<end)
			{
				continue;
			}

			if(i>=begin_A && i<end_A)
			{
				continue;
			}

			if(valid_I[i]>0)
			{
				// record the valid index
				index_I[out_i++]=i;
			}
		}
printf("check if count_I==out_i: %s\n", (count_I==out_i)? "true": "false");


		double * alpha_t = new double[count_ignored_A];
printf("first_round_l = %d \n", first_round_l);
		
		clock_t time_start = clock(), time_end;
		approximate_solution(prob, param, index_M, count_M, index_O, count_O, index_I, count_I, index_A, count_A, index_R, count_R, all_Q, whole_Alpha, perm, Global_Rho, alpha_t);
		time_end = clock();
		time_comsuming += (double)(time_end-time_start)/CLOCKS_PER_SEC;
		printf("elasped time for approximate_solution() is: %lfs\n", (double)(time_end-time_start)/CLOCKS_PER_SEC);
printf("---->alpha_t:\n");

		int greater = 0;
		int smaller = 0;
		int count_for_average = 0;

		double sum_a = 0;
		for (int i = 0; i < count_A; ++i)
		{
printf("i = %d alpha_t[i]= %lf\n", i, alpha_t[i]);
			// sum_a += prob->y[perm[index_A[i]]]*alpha_t[i];
			// method 1
			// if(alpha_t[i]<0&&prob->y[perm[index_A[i]]]<0)
			// {
			// 	greater++;
			// }
			// else if(alpha_t[i]<0&&prob->y[perm[index_A[i]]]>0)
			// {
			// 	smaller++;
			// }
			
			// method 2
			// if(alpha_t[i]<0)
			// {
			// 	alpha_t[i] = 0;
			// }
			// else if(alpha_t[i]>variable_C)
			// {
			// 	alpha_t[i]=variable_C;
			// }
			// sum_a += prob->y[perm[index_A[i]]]*alpha_t[i];

			// method 3
			if(alpha_t[i]<0||(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001))
			{
				alpha_t[i] = 0;
				count_for_average++;
			}
			else if(alpha_t[i]>variable_C||(alpha_t[i]>variable_C-0.0000001&&alpha_t[i]<variable_C+0.0000001))
			{
				alpha_t[i] = variable_C;
				count_for_average++;
			}
			sum_a += prob->y[perm[index_A[i]]]*alpha_t[i];

// printf("i = %d alpha_t[i]= %lf\n", i, alpha_t[i]);
		}

printf("after adjusting\n");
		for (int i = 0; i < count_A; ++i)
		{
printf("i = %d alpha_t[i]= %lf\n", i, alpha_t[i]);
		}

// printf("---->alpha_r:\n");
		// compare sum_r with sum_a, and then adjust alpha in St when sum_r is not equal to sum_a
		double sum_r = 0;
		for (int i = 0; i < count_R; ++i)
		{
// printf("i = %d alpha_t[i]= %lf\n", i, whole_Alpha[perm[index_R[i]]]);
			sum_r += prob->y[perm[index_R[i]]]*whole_Alpha[perm[index_R[i]]];
		}

printf("Compare sigma_St(A) vs. sigma_Sr(R): %lf vs. %lf\n", sum_a, sum_r);
///////////////////////////////////////////////////////////////

// printf("after adjusting and checkk\n");
// 		for (int i = 0; i < count_A; ++i)
// 		{
// printf("i = %d alpha_t[i]= %lf\n", i, alpha_t[i]);
// 		}

		if(sum_a>sum_r)
		{

			// double average = (double)(sum_a-sum_r)/greater;
			// double average = (double)(sum_a-sum_r)/count_A;
			double average = (double)(sum_a-sum_r)/(count_A-count_for_average);
printf("sum_a>sum_r, sum_a-sum_r = %lf greater = %d average = %lf count_A = %d count_for_average =%d\n", sum_a-sum_r, greater, average, count_A, count_for_average);
			int count_check = 0;
			for (int m = 0; m < count_A; ++m)
			{

// printf("at first: i = %d alpha_t vs. alpha_St: %lf vs. %lf\n", m, alpha_t[m], whole_Alpha[perm[index_A[m]]]);

				if(((alpha_t[m]>-0.0000001)&&(alpha_t[m]<0.0000001)))
				{
					whole_Alpha[perm[index_A[m]]] = 0;
					count_check++;
					continue;
				}

				if((alpha_t[m]>variable_C-0.0000001)&&(alpha_t[m]<variable_C+0.0000001))
				{
					whole_Alpha[perm[index_A[m]]] = variable_C;
					count_check++;
					continue;
				}

// printf("second: i = %d alpha_t vs. alpha_St: %lf vs. %lf\n", m, alpha_t[m], whole_Alpha[perm[index_A[m]]]);

				if(prob->y[perm[index_A[m]]]<0)
				{
					whole_Alpha[perm[index_A[m]]] = alpha_t[m] + average;
					greater++;
				}
				else
				{
					whole_Alpha[perm[index_A[m]]] = alpha_t[m] - average;
					greater++;
				}
				
// printf("third: i = %d alpha_t vs. alpha_St: %lf vs. %lf\n", m, alpha_t[m], whole_Alpha[index_A[m]]);
				// if(alpha_t[m]<0 && prob->y[perm[index_A[m]]]<0)
				// {
				// 	whole_Alpha[perm[index_A[m]]] = alpha_t[m] + average;
				// 	count_check++;
				// }
				// else
				// {
				// 	whole_Alpha[perm[index_A[m]]] = alpha_t[m];
				// }
			}
printf("if greater is equal to count_check: %d vs. %d sum = %d \n", greater, count_check, greater+count_check );
		}
		else
		{
			// sum_a<sum_r
			// double average = (double)(sum_r-sum_a)/smaller;
			// double average = (double)(sum_r-sum_a)/count_A;
			double average = (double)(sum_r-sum_a)/(count_A-count_for_average);
printf("sum_a<sum_r, sum_r-sum_a = %lf smaller = %d average = %lf count_A = %d count_for_average = %d\n", sum_r-sum_a, smaller, average, count_A, count_for_average);
			int count_check = 0;
			for (int m = 0; m < count_A; ++m)
			{
				if(((alpha_t[m]>-0.0000001)&&(alpha_t[m]<0.0000001)))
				{
					whole_Alpha[perm[index_A[m]]] = 0;
					count_check++;
					continue;
				}

				if((alpha_t[m]>variable_C-0.0000001)&&(alpha_t[m]<variable_C+0.0000001))
				{
					whole_Alpha[perm[index_A[m]]] = variable_C;
					count_check++;
					continue;
				}
				// if(((alpha_t[m]>-0.0000001)&&(alpha_t[m]<0.0000001))||((alpha_t[m]>variable_C-0.0000001)&&(alpha_t[m]<variable_C+0.0000001)))
				// {
				// 	count_check++;
				// 	continue;
				// }
// printf("i = %d alpha_t vs. alpha_St: %lf vs. %lf\n", m, alpha_t[m], whole_Alpha[index_A[m]]);
				if(prob->y[perm[index_A[m]]]<0)
				{
					whole_Alpha[perm[index_A[m]]] = alpha_t[m] - average;	
					smaller++;
				}
				else
				{
					whole_Alpha[perm[index_A[m]]] = alpha_t[m] + average;
					smaller++;
				}
printf("i = %d alpha_t vs. alpha_St: %lf vs. %lf\n", m, alpha_t[m], whole_Alpha[index_A[m]]);
				// if(alpha_t[m]<0 && prob->y[perm[index_A[m]]]>0)
				// {
				// 	whole_Alpha[perm[index_A[m]]] = alpha_t[m] + average;
				// 	count_check++;
				// }
				// else
				// {
				// 	whole_Alpha[perm[index_A[m]]] = alpha_t[m];
				// }
			}
printf("if smaller is equal to count_check: %d vs. %d sum = %d \n", smaller, count_check, smaller+count_check);
		}

		sum_a = 0;
		for (int i = 0; i < count_A; ++i)
		{
			sum_a += prob->y[perm[index_A[i]]]*whole_Alpha[perm[index_A[i]]];
		}

		sum_r = 0;
		for (int i = 0; i < count_R; ++i)
		{
			sum_r += prob->y[perm[index_R[i]]]*whole_Alpha[perm[index_R[i]]];
		}
printf("Second comparing sigma_St(A) vs. sigma_Sr(R): %lf vs. %lf\n", sum_a, sum_r);

		// ======================================= construct svm_model
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
		
		double * subalpha = new double[subprob.l];

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			// XXX
			subalpha[k] = whole_Alpha[perm[j]];
// printf("i = %d \t perm[i] = %d \t prob->y[perm[j]] = %lf \t whole_Alpha[perm[j]] = %lf\n", j, perm[j], prob->y[perm[j]], whole_Alpha[perm[j]]);
			// subalpha[k] = 3;
			++k;
		}

		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			// XXX
			subalpha[k] = whole_Alpha[perm[j]];
// printf("i = %d \t perm[i] = %d \t prob->y[perm[j]] = %lf \t whole_Alpha[perm[j]] = %lf\n", j, perm[j], prob->y[perm[j]], whole_Alpha[perm[j]]);
			// subalpha[k] = 3;
			++k;
		}
		
// 		// check sum{alpha_i * y_i}
// 		double sum_y_a = 0;
// 		for (int i = 0; i < subprob.l; ++i)
// 		{
// 			sum_y_a += subprob.y[i]*subalpha[i];
// 		}
// printf("sum{alpha_i * y_i} = %lf\n", sum_y_a);

		struct svm_model *submodel;	  // here we don't allocate the memory, in build_model we do.
		// submodel =  svm_train(&subprob, param);

		clock_t start_train = clock(), end_train;
		submodel =  svm_train_origin(&subprob, param, subalpha);
		end_train = clock();
		time_comsuming_train += (double)(end_train-start_train)/CLOCKS_PER_SEC;
		printf("elasped time for svm_train_origin() is: %lfs, current fold is: %d\n", (double)(end_train-start_train)/CLOCKS_PER_SEC, cur_fold+1);


		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
			{
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
// printf("i = %d \t perm[i] %d \t target vs. origin: %lf vs. %lf\n", j, perm[j], target[perm[j]], prob->y[perm[j]]);
			}
			free(prob_estimates);
		}
		else
			for(j=begin;j<end;j++)
			{
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
				// printf("i = %d \t perm[i] %d \t target vs. origin: %lf vs. %lf\n", j, perm[j], target[perm[j]], prob->y[perm[j]]);
			}
		delete[] alpha_t;
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
		// =========================================

	}

printf("\ntime consuming for approximate_solution() in all k fold: %lfs\n", time_comsuming);
printf("time consuming for solve() in all k fold: %lfs\n", time_comsuming_solve);  
printf("time consuming for solve_c() in all k fold: %lfs\n", time_comsuming_solve_c);
printf("time consuming for svm_train_one_origin() in all k fold: %lfs\n", replacement_svm_train_time);
printf("time consuming for svm_train() in all k fold: %lf\n\n", time_comsuming_train);
printf("iterations_check = %d\n", iterations_check);
printf("iterations_replacement = %d\n", iterations_replacement);
printf("iterations_libsvm = %d\n", iterations_libsvm);
printf("replacement_svm_train_time = %lfs\n", replacement_svm_train_time);
printf("libsvm_svm_train_time = %lfs\n", libsvm_svm_train_time);


	free(fold_start);
	free(perm);

	for(int i=0;i<l;i++)
	{
		free(all_Q[i]);
	}
	delete[] whole_Alpha;
}

// Stratified cross validation
void svm_cross_validation_libsvm(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}
	double time_comsuming_train_libsvm = 0;
	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
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
		// modify
		clock_t start_train_libsvm = clock(), end_train_libsvm;

		struct svm_model *submodel = svm_train(&subprob,param);

		end_train_libsvm = clock();
		time_comsuming_train_libsvm+=(double)(end_train_libsvm-start_train_libsvm)/CLOCKS_PER_SEC;
		printf("elasped time for svm_train() is: %lfs, current fold is: %d \n", (double)(end_train_libsvm-start_train_libsvm)/CLOCKS_PER_SEC, i);
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	printf("time consuming for svm_train() in all k fold: %lf\n", time_comsuming_train_libsvm);
	printf("iterations_check = %d\n", iterations_check);
	free(fold_start);
	free(perm);
}

int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices)
{
	if (model->sv_indices != NULL)
		for(int i=0;i<model->l;i++)
			indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model)
{
	return model->l;
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
		int nr_class = model->nr_class;
		int l = model->l;
		
		double *kvalue = Malloc(double,l);
		// struct svm_node
		// {
		// 	int index;
		// 	double value;
		// };
		for(i=0;i<l;i++)
		{
// printf("i = %d model->SV[i].index = %d model->SV[i].value = %lf ", i, model->SV[i]->index, model->SV[i]->value);
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);
// printf("kvalue[i] = %lf\n", kvalue[i]);
		}

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
// printf("coef1 = %lf coef2 = %lf\n", coef1[0], coef2[0]);
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
// printf("sum += coef1[si+k] * kvalue[si+k] = %lf\n", sum);
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
// printf("model->rho[p] = %lf sum before vs. after: %lf vs. ", model->rho[p], sum);
				sum -= model->rho[p];
// printf("%lf\n", sum);
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
// printf("i = %d vote_max_idx = %d vote[i] = %d vote[vote_max_idx] = %d model->label[i]~model->label[%d] = %d vs. model->label[vote_max_idx] = %d which the bigger one: ", i, vote_max_idx, vote[i], vote[vote_max_idx], i, model->label[i], model->label[vote_max_idx]);
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;
// printf(" %d model->label[%d] = %d\n", vote_max_idx, vote_max_idx, model->label[vote_max_idx]);
		}

		free(kvalue);
		free(start);
		free(vote);
// printf("vote_max_idx = %d return model->label[vote_max_idx]~model->label[%d] = %d\n", vote_max_idx, vote_max_idx, model->label[vote_max_idx]);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else 
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	}
	else 
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, svm_model* model)
{
	svm_parameter& param = model->param;
	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");	
				return false;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			FSCANF(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			FSCANF(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			FSCANF(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			FSCANF(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			FSCANF(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return false;
		}
	}

	return true;

}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model,1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;
	
	// read header
	if (!read_model_header(fp, model))
	{
		fprintf(stderr, "ERROR: fscanf failed to read model\n");
		setlocale(LC_ALL, old_locale);
		free(old_locale);
		free(model->rho);
		free(model->label);
		free(model->nSV);
		free(model);
		return NULL;
	}
	
	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label= NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB= NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}


// other version of getting the value of f(x_i)
void get_Qij(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_Q)
{
	switch(param->svm_type)
	{
		case C_SVC:
		{
			int l=prob->l;
			schar *y = new schar[l];
			
        // printf("in solve_c_svc\n");             
        // original order of raw data             
        for(int i=0;i<l;i++)             
        { // 
        	// printf("get_Qijget_Qij the index is: %d and the y[%d] is: %lf\n", i, i,prob->y[i]);
// alpha[i] = 0;                 // minus_ones[i] = -1;                 
// used perm when call svm_train function, no need to use perm again here
			if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;             
			}             
			//SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
// SVC_Q* svc_q = new SVC_Q(*prob, *param, y);
			
			// for (int i = 0; i < l; ++i)
			// {
			// 	all_Q[i] = svc_q->get_Q(i,l);
			// }

			for (int i = 0; i < l; ++i)
			{
				for (int j = 0; j < l; ++j)
				{
					// printf("all_Q[%d][%d]=%f\n", i, j, all_Q[i][j]);
					// data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));   k_function(prob->x[xi], prob->x[j], *param)
					all_Q[i][j] = (Qfloat)(y[i]*y[j]*kernel_function(prob->x[i], prob->x[j], *param));
				}
			}
			// free(svc_q);
			// delete svc_q;
			delete[] y;
		}
			// solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			// solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			// solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			// solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			// solve_nu_svr(prob,param,alpha,&si);
			break;
	}
}



double get_Fxi(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *whole_Alpha, double rho)
{
	// xi must be permutated
	double sigma=0;
	switch(param->svm_type)
	{
		case C_SVC:
		{
			int l=prob->l;
			schar *y = new schar[l];
			
		// printf("in solve_c_svc\n");
			// original order of raw data
			for(int i=0;i<l;i++)
			{
// printf("get_Qijget_Qij the index is: %d and the y[%d] is: %lf\n", i, i,prob->y[i]);	
				// alpha[i] = 0;
				// minus_ones[i] = -1;
				// used perm when call svm_train function, no need to use perm again here
				if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
			}

			// SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
			// SVC_Q svc_q = SVC_Q(*prob, *param, y);
			// Kernel *kernel = new Kernel(l, prob->x, *param);

			// double Kernel::k_function(const svm_node *x, const svm_node *y,
			//   const svm_parameter& param)
			// Kernel(int l, svm_node * const * x, const svm_parameter& param);
			for (int j = 0; j < l; ++j)
			{
// printf("test kernel_function(%d, %d), myfunction: %lf, k_function: %lf\n", xi, j, svc_q->k_function(prob->x[xi], prob->x[j], *param), powi(param->gamma*Kernel_dot(prob->x[xi], prob->x[j])+param->coef0,param->degree));				
				sigma += whole_Alpha[j]*prob->y[j]*(kernel_function(prob->x[xi], prob->x[j], *param));
				// (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
			}
			// free(svc_q);
			delete[] y;
		}
			// solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			// solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			// solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			// solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			// solve_nu_svr(prob,param,alpha,&si);
			break;
	}
// printf("sigma sigma = %lf\n", sigma);
	return sigma + rho;
}




double kernel_function(const svm_node *x, const svm_node *y, const svm_parameter& param)
{
// test: bean
// printf("the kernel_type is %d\n", param.kernel_type);

	switch(param.kernel_type)
	{
		case LINEAR:
			return Kernel_dot(x,y);
		case POLY:
			{
// printf("get in POLY\n");
				return powi(param.gamma*Kernel_dot(x,y)+param.coef0,param.degree);
			}
		case RBF:
		{
// printf("get in RBF\n");
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*Kernel_dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}
// get the value of f(x_i)
double decision_function_x(const struct svm_node *xi, double *alpha, double rho, const struct svm_problem *prob, const struct svm_parameter *param, int *perm)
{
	double sigma=0;

	// polynomial kernel: powi(gamma*dot(x[i],x[j])+coef0,degree) 
	for(int j=0;j<prob->l;j++)
	{
		sigma+=alpha[perm[j]]*prob->y[perm[j]]*powi(param->gamma*Kernel_dot(xi, prob->x[perm[j]])+param->coef0,param->degree);
	}

	return sigma+rho;
}

// data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
// get all Q_ij
void Q_ij(const struct svm_problem *prob, const struct svm_parameter *param, int *perm, Qfloat** all_Q)
{
	
	for(int m=0;m<prob->l;m++)
		for(int n=0;n<prob->l;n++)
		{
			all_Q[m][n]=(Qfloat)(prob->y[perm[m]]*prob->y[perm[n]]*(powi(param->gamma*Kernel_dot(prob->x[perm[m]], prob->x[perm[n]])+param->coef0,param->degree)));
		}

}

// calculate the phi in equetion 10
void calculate_first_phi(const struct svm_problem *prob, Qfloat **Q_ij, int *index_M, int count_M,  int *index_A, int count_A, int *index_R, int count_R, double *alpha, double C, int *perm, double *first_phi)
{	// prob is original
printf("\n>>>>>>>> get into calculate_first_phi\n");
printf("count_M=%d\n", count_M);
printf("count_A=%d\n", count_A);
printf("count_R=%d\n", count_R);
	mat first_matrix(count_M+1, count_M+1);
// printf("cccccccccccccccccccccccccc\n");
printf("first_matrix size is : %d x %d \n", count_M+1, count_M+1);
	mat second_matrix(count_M+1,count_A+count_R);
printf("second_matrix size is : %d x %d \n", count_M+1, count_A+count_R);
	mat third_matrix(count_A+count_R,1);
printf("third_matrix size is : %d x %d \n", count_A+count_R, 1);
	mat inv_first_matrix(count_M+1, count_M+1);
printf("inv_first_matrix size is : %d x %d \n", count_M+1, count_M+1);
	mat result(count_M+1, 1);
printf("result size is : %d x %d \n", count_M+1, 1);
	// construct the first matrix
	// set first_matrix[0][0]
	first_matrix(0,0) = 0;

	// set first row except for first_matrix[0][0] adn first colomn except for first_matrix[0][0]
	for(int c=0;c<count_M;c++)
	{
		first_matrix(0,c+1)=prob->y[perm[index_M[c]]];
		first_matrix(c+1,0)=prob->y[perm[index_M[c]]];
	}
// printf("atttttttttttt\n");
	// set the rest of first_matrix
	for(int i=0;i<count_M;i++)
	{
		for (int j = 0; j < count_M; ++j)
		{
			// printf("atttttttttttt\n");
			// printf("index_M[%d]=%d  \t index_M[%d]=%d \n", i,index_M[i],j,index_M[j]);
			// printf("index_M[%d] index_M[%d] Q_ij[index_M[%d]][index_M[%d]]=%f\n", i,j,index_M[i],index_M[j],Q_ij[index_M[i]][index_M[j]]);
			first_matrix(i+1,j+1)= Q_ij[perm[index_M[i]]][perm[index_M[j]]];
			// printf("first_matrix(%d, %d)=%lf\n", i+1,j+1,first_matrix(i+1,j+1));
		}
	}

	// construct the second matrix
	// set first row
	for(int c=0;c<count_A;c++)
	{
		second_matrix(0,c) = prob->y[perm[index_A[c]]];
	}

// printf("count_R=%d\n", count_R);

	for(int c=0;c<count_R;c++)
	{
		// printf("index_R[%d] = %d\n", c, index_R[c]);
		second_matrix(0,count_A+c) = prob->y[perm[index_R[c]]];
	}

	// set the rest of second_matrix
printf("count_M before second_matrix : %d\n", count_M);
printf("count_M + count_R : %d\n", count_M+count_R);
	for(int r=0;r<count_M;r++)
	{
// printf("row in every loop: %d\n", r);
		for(int c=0;c<count_A;c++)
		{	
			// TODO
			// printf("one index_M[%d]=%d, index_A[%d]=%d, Q[%d][%d]=%lf\n", r, index_M[r], c, index_A[c], index_M[r],index_A[c], Q_ij[index_M[r]][index_A[c]]);
			second_matrix(r+1,c) = Q_ij[perm[index_M[r]]][perm[index_A[c]]];
		}

		for(int c=0;c<count_R;c++)
		{
			// TODO
			// printf("two index_M[%d]=%d, index_R[%d]=%d\n", r, index_M[r], c, index_R[c]);
// printf("two index_M[%d]=%d, index_R[%d]=%d, Q[%d][%d]=%lf\n", r, index_M[r], c, index_R[c], index_M[r], index_R[c], Q_ij[index_M[r]][index_R[c]]);

			second_matrix(r+1,count_A+c) = Q_ij[perm[index_M[r]]][perm[index_R[c]]];
		}
	}
// printf("check index_m[%d]=%d index_A[%d]=%d Q_ij[%d][%d] = %lf \n", count_M-1, index_M[count_M-1], count_R-1, index_R[count_R-1], index_M[count_M-1], index_R[count_R-1], Q_ij[index_M[count_M-1]][index_R[count_R-1]]);
	
	// construct the third matrix
	for(int r=0;r<count_A;r++)
	{
		third_matrix(r,0)=C-alpha[perm[index_A[r]]];
// printf("index_A[%d] = %d \t C = %lf alpha[index_A[%d]] = %lf\n", r, index_A[r], C, r, alpha[index_A[r]]);
	}

	for(int r=0;r<count_R;r++)
	{
		third_matrix(count_A+r,0)=-alpha[perm[index_R[r]]];
// printf("index_R[%d] = %d \t -alpha[index_R[%d]] = %lf\n", r, index_R[r], r, -alpha[index_R[r]]);
	}

// printf("----------first_matrix\n" );
// first_matrix.print();


// check out if there are two same instance
// bool same = true;
// for (int i = 0; i < count_M+1; ++i)
// {
// 	for (int j = i+1; j < count_M+1; ++j)
// 	{
// 		for (int m = 0; m < count_M+1; ++m)
// 		{
// // if((first_matrix(i,m) >= first_matrix(j,m) - 0.000001)&&(first_matrix(i,m) <= first_matrix(j,m) + 0.000001))
// // {
// 	// printf("i=%d, j=%d, m=%d, first_matrix(i,m):%lf vs. first_matrix(j,m):%lf\n", i, j, m, first_matrix(i,m), first_matrix(j,m));	
// // }

// 			if(!((first_matrix(i,m) >= first_matrix(j,m) - 0.000001)&&(first_matrix(i,m) <= first_matrix(j,m) + 0.000001)))
// 			{
// 				same =false;
// 			}
// 			// else
// 			// {
// 			// 	same = false;
// 			// }
// 		}
// 		if(same)
// 		{
// printf("check check check they are same when i=%d j=%d\n", i, j);
// 		}
// 	}
// }

// printf("check if there is two same lines: %d \n", same);
	inv_first_matrix = pinv(first_matrix);
	// inv_first_matrix = inv(first_matrix);

// printf("----------inv_first_matrix\n" );
// inv_first_matrix.print();

// printf("----------second_matrix\n" );
// second_matrix.print();

// printf("----------third_matrix\n" );
// third_matrix.print();


	result = -inv_first_matrix*second_matrix*third_matrix;

// printf("==========result\n" );
// result.print();

	for(int i=0;i<count_M+1;i++)
	{
		first_phi[i]=result(i,0);
// printf("i = %d\t first_phi[%d] = %lf\n", i, i, first_phi[i]);
	}
printf("<<<<<<<< get out of calculate_first_phi\n\n");
}

// calculate the Phi in equation 11
void calculate_second_phi(const struct svm_problem *prob, Qfloat **Q_ij, int *index_M, int count_M,  int *index_A, int count_A, int *index_R, int count_R, double *alpha, double C, int *perm, double* first_phi, double *second_phi)
{
printf("\n>>>>>>>>> get into calculate_second_phi\n");
	mat previous_first_phi(count_M+1,1);
	mat first_matrix(prob->l, count_M+1);
	
	mat second_matrix(prob->l,count_A);
	mat third_matrix(count_A,1);

	mat fourth_matrix(prob->l, count_R);
	mat fifth_matrix(count_R, 1);

	mat result(prob->l, 1);

	// first_phi, array to mat
	for (int i = 0; i < count_M+1; ++i)
	{
		previous_first_phi(i,0) = first_phi[i];
	}

	// construct the first matrix
	// set first row
	for(int r=0;r<prob->l;r++)
	{
		first_matrix(r,0)=prob->y[r];
	}
// printf("tttttttttttt\n");
	// set rest of first matrix
	for (int i = 0; i < prob->l; ++i)
	{
		for (int j = 0; j < count_M; ++j)
		{
			// first_matrix(i,j+1)=Q_ij[i][index_M[j]];
			first_matrix(i,j+1)=Q_ij[i][perm[index_M[j]]];
		}
	}

	// construct the second matrix
	for (int i = 0; i < prob->l; ++i)
	{
		for (int j = 0; j < count_A; ++j)
		{
			second_matrix(i,j)=Q_ij[i][perm[index_A[j]]];
		}
	}
	
	// construct the third matrix
	for (int i = 0; i < count_A; ++i)
	{
		third_matrix(i,0)=C-alpha[perm[index_A[i]]];
	}

	// construct the fourth matrix
	for (int i = 0; i < prob->l; ++i)
	{
		for (int j = 0; j < count_R; ++j)
		{
			fourth_matrix(i,j)=Q_ij[i][perm[index_R[j]]];
		}
	}

	//construct the fifth matrix
	for (int i = 0; i < count_R; ++i)
	{
		fifth_matrix(i,0)=alpha[perm[index_R[i]]];
	}

// printf("first_matrix: ==========result\n" );
// first_matrix.print();

// printf("previous_first_phi: ==========result\n" );
// previous_first_phi.print();

// printf("second_matrix: ==========result\n" );
// second_matrix.print();

// printf("third_matrix: ==========result\n" );
// third_matrix.print();

// printf("fourth_matrix: ==========result\n" );
// fourth_matrix.print();

// printf("fifth_matrix: ==========result\n" );
// fifth_matrix.print();

//////////////////////////////////////////////////////


// result1 is largercc
// mat result1(prob->l, 1);
// mat result2(prob->l, 1);
// mat result3(prob->l, 1);


// printf("first_matrix: ==========result\n" );
// first_matrix.print();

// printf("previous_first_phi: ==========result\n" );
// previous_first_phi.print();

// result1 = first_matrix*previous_first_phi;
// result2 = second_matrix*third_matrix;
// result3 = fourth_matrix*fifth_matrix;

// printf("result1: ==========result\n" );
// result1.print();

// printf("result2: ==========result\n" );
// result2.print();

// printf("result3: ==========result\n" );
// result3.print();

	result=first_matrix*previous_first_phi + second_matrix*third_matrix-fourth_matrix*fifth_matrix;

	// printf("second_phi: ==========result\n" );
	// result.print();

	for(int i=0;i<prob->l;i++)
	{
		second_phi[i]=result(i,0);
// printf("i=%d \t second_phi[%d] = %lf \n", i, i, second_phi[i]);
	}
printf("<<<<<<<<< get out of calculate_second_phi\n\n");
}


// use minimal eta to update alpha
void update_alpha(double eta_min, double* whole_Alpha, double* first_phi, double delta_b, int count_M, int* index_M, int count_R, int* valid_R, int* index_R, int count_A, int* index_A, int begin, int* perm)
{
	// ======================================================== update
	// update corresponding Sets
	// update equation 10
	// double delta_b = eta_min*first_phi[0];
printf(">>>>>> get into update_alpha\n");
printf("delta_b = %lf\n", delta_b);
	double* delta_alpha_M = new double[count_M];

	// update Global_Rho
	// b = -rho
	Global_Rho = Global_Rho + delta_b;

	// update alpha in M
	for (int i = 0; i < count_M; ++i)
	{
		if(eta_min >= -0.000001 && eta_min <= 0.000001)
		{
			delta_alpha_M[i]=0*first_phi[i+1];
		}
		else
		{
			delta_alpha_M[i]=eta_min*first_phi[i+1];
		}
				
printf("in Set M: i=%d, eta_min=%lf, first_phi[i+1]=%lf, eta_min*first_phi[i+1] = %lf \n", i, eta_min, first_phi[i+1], delta_alpha_M[i]);
printf("in Set M: i=%d, whole_Alpha[%d], %lf vs. ", i, index_M[i], whole_Alpha[index_M[i]]);
		whole_Alpha[index_M[i]]=whole_Alpha[index_M[i]]+delta_alpha_M[i];
printf("%lf\n", whole_Alpha[index_M[i]]);
	}

	// update alpha in R
	// update_count_R = count_R;
printf("before updating count_R, the count_R is %d\n", count_R);
	for (int i = 0; i < count_R; ++i)
	{
printf("in Set R: i=%d, whole_Alpha[%d], %lf vs. ", i, index_R[i], whole_Alpha[index_R[i]]);
		// Conditional jump or move depends on uninitialised value(s)
		if(eta_min >= -0.000001 && eta_min <= 0.000001)
		{
			whole_Alpha[perm[index_R[i]]]=whole_Alpha[perm[index_R[i]]]*(1 - 0);
		}
		else
		{
			whole_Alpha[perm[index_R[i]]]=whole_Alpha[perm[index_R[i]]]*(1 - eta_min);
		}
printf("%lf\n", whole_Alpha[index_R[i]]);
		if(whole_Alpha[perm[index_R[i]]]<=0)
		{
			valid_R[index_R[i]-begin] = 0;
			// index should be crrected XXX
			// update_count_R--;
			count_R--;
		}
				
	}
	printf("at last count_R = %d\n", count_R);

	// update alpha in A
	for (int i = 0; i < count_A; ++i)
	{
printf("in Set A: i=%d, whole_Alpha[%d], %lf vs. ", i, index_A[i], whole_Alpha[index_A[i]]);
		// Conditional jump or move depends on uninitialised value(s)
		if(eta_min >= -0.000001 && eta_min <= 0.000001)
		{
			whole_Alpha[index_A[i]]=whole_Alpha[index_A[i]]+0*(variable_C-whole_Alpha[index_A[i]]);
		}
		else
		{
			whole_Alpha[index_A[i]]=whole_Alpha[index_A[i]]+eta_min*(variable_C-whole_Alpha[index_A[i]]);
		}
printf("%lf\n", whole_Alpha[index_A[i]]);
	}

printf("final eta_min=%lf\n", eta_min);
printf("<<<<<<< get out of update_alpha\n");
}


void approximate_solution(const struct svm_problem *prob, const struct svm_parameter *param, int *index_M, int count_M, int *index_O, int count_O, int *index_I, int count_I, int *index_A, int count_A, int *index_R, int count_R, Qfloat **Q_ij, double *alpha, int *perm, double rho, double* alpha_St)
{
printf(">>>>>>>>>>> get into approximate_solution\n");
	
	int l = prob->l;
	mat left_origin(l+1, count_A);
	mat left_T(count_A, l+1);
	mat left_multiple(count_A, count_A);
	mat left_multiple_inv(count_A, count_A);
	mat second_matrix(l+1, 1);
	mat up_sum(l, 1);
	mat down_multiple(1, 1);
	mat Q_n_Sr(l, count_R);
	mat alpha_Sr(count_R, 1);
	mat result(count_A, 1);
	mat y_T(1, count_R);
	mat Q_alpha(l, 1);

	double  *delta_f = new double[l];

	// print testing
printf("count_M = %d\n", count_M);
printf("count_O = %d\n", count_O);
printf("count_I = %d\n", count_I);
printf("count_A = %d\n", count_A);
printf("count_R = %d\n", count_R);

printf("count_ignored_R+count_ignored_A+count_M+count_O+count_I = %d\n", count_R+count_A+count_M+count_O+count_I);
	
	// calculate the first term in equation (8) on the left
	for (int i = 0; i < l; ++i)
	{
		for (int j = 0; j < count_A; ++j)
		{
// printf("j = %d \n", j);
// printf("index_A[j] = %d \n", index_A[j]);
// printf("perm[index_A[j]] = %d \n", perm[index_A[j]]);
			left_origin(i, j) = Q_ij[i][perm[index_A[j]]];
		}
	}


	for (int i = 0; i < count_A; ++i)
	{
		left_origin(l, i) = prob->y[perm[index_A[i]]];
	}

	// calculate the delta_f in the second term
	// M
	for (int i = 0; i < count_M; ++i)
	{
		delta_f[perm[index_M[i]]] = 0;
	}

	// O
	for (int i = 0; i < count_O; ++i)
	{
		delta_f[perm[index_O[i]]] = prob->y[perm[index_O[i]]] - get_Fxi(perm[index_O[i]], prob, param, alpha, rho);
	}

	// I
	for (int i = 0; i < count_I; ++i)
	{
		delta_f[perm[index_I[i]]] = prob->y[perm[index_I[i]]] - get_Fxi(perm[index_I[i]], prob, param, alpha, rho);
	}

	// A
	for (int i = 0; i < count_A; ++i)
	{
		delta_f[perm[index_A[i]]] = 0;
	}

printf("--------------------------\n");

	// Q:Sr & y_T
	for (int i = 0; i < count_R; ++i)
	{
// printf("i = %d\n", i);
// printf("index_R[i] = %d\n", index_R[i]);
// printf("perm[index_R[i]] = %d\n", perm[index_R[i]]);
		alpha_Sr(i, 0) = alpha[perm[index_R[i]]];
// printf("alpha[perm[index_R[i]]] = %lf\n", alpha[perm[index_R[i]]]);
// printf("alpha_Sr(i, 1) = %lf\n", alpha_Sr(i, 0));
		y_T(0, i) = prob->y[perm[index_R[i]]];
// printf("y_T(1, i) = %lf\n", y_T(0, i));
	}

	// alpha_Sr
	for (int i = 0; i < l; ++i)
	{
		for (int j = 0; j < count_R; ++j)
		{
			Q_n_Sr(i, j) = Q_ij[i][perm[index_R[j]]];
		}
	}

	up_sum = Q_n_Sr*alpha_Sr;

	for (int i = 0; i < l; ++i)
	{
		second_matrix(i, 0) = prob->y[i]*delta_f[i] + up_sum(i, 0);
	}

	down_multiple = y_T * alpha_Sr;
// printf("test down_multiple = %lf vs. %lf \n", down_multiple(0, 0), -1*down_multiple(0, 0));
	second_matrix(l, 0) = -1*down_multiple(0, 0);

// printf("ttttttttttttttttttttttt\n");

// printf("left_origin\n");
// for (int i = 0; i < l+1; ++i)
// {
// 	for (int j = 0; j < count_A; ++j)
// 	{
// printf("%lf \t", left_origin(i, j));
// 	}
// printf("\n");
// }

	left_T = trans(left_origin);

// printf("left_T\n");
// for (int i = 0; i < count_A; ++i)
// {
// 	for (int j = 0; j < l+1; ++j)
// 	{
// printf("%lf \t", left_T(i, j));
// 	}
// printf("\n");
// }

	left_multiple = left_T*left_origin;

// printf("left_multiple:\n");
// for (int i = 0; i < count_A; ++i)
// {
// 	for (int j = 0; j < count_A; ++j)
// 	{
// printf("%lf \t", left_multiple(i, j));
// 	}
// printf("\n");
// }

	left_multiple_inv = inv(left_multiple);

	result = left_multiple_inv*left_T*second_matrix;

// printf("result\n");
	for (int i = 0; i < count_A; ++i)
	{
		alpha_St[i] = result(i, 0);
// printf("%lf \t", alpha_St[i]);
	}
// printf("\n");

printf("<<<<<<<<<<< get out of approximate_solution\n");
}


//
// Interface functions
//
svm_model *svm_train_origin(const svm_problem *prob, const svm_parameter *param, double * whole_Alpha)
{
printf(">>>>>>>> get into svm_train_origin\n");
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX
	
	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		if(nr_class == 1) 
			info("WARNING: training data in only one class. See README for details.\n");
		
		svm_node **x = Malloc(svm_node *,l);
		int i;
		double *alpha=new double[l];
		for(i=0;i<l;i++)
		{
			x[i] = prob->x[perm[i]];
			alpha[i]=whole_Alpha[perm[i]];

		}
		

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);
		

		// check alpha
		// decision_function *f1 = Malloc(decision_function,nr_class*(nr_class-1)/2);
		

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				double * subalpha = new double[sub_prob.l];
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
					subalpha[k]=alpha[si+k];
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
					subalpha[ci+k] = alpha[sj+k];
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

double sum_y_a = 0;
		for (int m = 0; m < prob->l; ++m)
		{
			sum_y_a += prob->y[m]*whole_Alpha[m];;
		}
printf("in svm_train_origin sum{alpha_i * y_i} = %lf\n", sum_y_a);


				clock_t svm_train_start, svm_train_end;

				// svm_train_start = clock();
				// check alpha difference
// 				f1[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
// 				svm_train_end = clock();
// iterations_libsvm +=iterations_check;
// printf("iterations_check = %d iterations_libsvm =%d elapsed time: %lf\n", iterations_check, iterations_libsvm, (double)(svm_train_end-svm_train_start)/CLOCKS_PER_SEC);
// libsvm_svm_train_time += (double)(svm_train_end-svm_train_start)/CLOCKS_PER_SEC;
	
				svm_train_start = clock();
				f[p] = svm_train_one_origin(&sub_prob,param,weighted_C[i],weighted_C[j], subalpha);
				svm_train_end = clock();

iterations_replacement +=iterations_check;
printf("iterations_check = %d iterations_replacement =%d elapsed time: %lf \n", iterations_check, iterations_replacement, (double)(svm_train_end-svm_train_start)/CLOCKS_PER_SEC);
replacement_svm_train_time += (double)(svm_train_end-svm_train_start)/CLOCKS_PER_SEC;

// 				svm_train_start = clock();
// 				// check alpha difference
// 				f1[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
// 				svm_train_end = clock();
// iterations_libsvm +=iterations_check;
// printf("iterations_check = %d iterations_libsvm =%d elapsed time: %lf\n", iterations_check, iterations_libsvm, (double)(svm_train_end-svm_train_start)/CLOCKS_PER_SEC);
// libsvm_svm_train_time += (double)(svm_train_end-svm_train_start)/CLOCKS_PER_SEC;

// printf("alpha difference: \n" );
// 				for (int i = 0; i < prob->l; ++i)
// 				{
// printf("libsvm vs. my_implement: %lf vs. %lf\n", f1[p].alpha[i], f[p].alpha[i]);
// 				}

				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				// delete[] subalpha; 
				free(sub_prob.x);
				free(sub_prob.y);
				++p;

			}

		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		delete [] alpha;
		free(x);
		free(weighted_C);
		free(nonzero);

		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);

		free(f);
		free(nz_count);
		free(nz_start);
		// delete[] subalpha;
	}
printf("<<<<<<<< get out of svm_train_origin\n");
	return model;
}