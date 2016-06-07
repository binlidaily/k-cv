#include <math.h>
#include <stdio.h>
#include <float.h>
// #include <armadillo>
#include <Eigen/Dense>

#include "rpi.h"
#include "svm.h"

// using namespace arma;
using namespace Eigen;

// #define DBL_MAX HUGE_VAL

double tol_equal1 = 0.000001;


typedef signed char schar;

double pow(double base, int times)
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
double Kernel_dot(const svm_node *px, const svm_node *py)
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

// calculate kernal value
double kernel_function(const svm_node *x, const svm_node *y, const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return Kernel_dot(x,y);
		case POLY:
			{
				return pow(param.gamma*Kernel_dot(x,y)+param.coef0,param.degree);
			}
		case RBF:
		{
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

// calculate Q_ij
void calculate_Q(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_Q)
{
	switch(param->svm_type)
	{
		case C_SVC:
		{
			int l=prob->l;
			schar *y = new schar[l];
			            
        	for(int i=0;i<l;i++)             
        	{
				if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;             
			}   

			for (int i = 0; i < l; ++i)
			{
				for (int j = 0; j < l; ++j)
				{
					all_Q[i][j] = (Qfloat)(y[i]*y[j]*kernel_function(prob->x[i], prob->x[j], *param));
				}
			}
			
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

// calculate Q_ij
void calculate_Kernel(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_K)
{
	switch(param->svm_type)
	{
		case C_SVC:
		{
			int l=prob->l;
			schar *y = new schar[l];
			            
        	for(int i=0;i<l;i++)             
        	{
				if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;             
			}   

			for (int i = 0; i < l; ++i)
			{
				for (int j = 0; j < l; ++j)
				{
					all_K[i][j] = (Qfloat)(kernel_function(prob->x[i], prob->x[j], *param));
				}
			}
			
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

double calculate_fx(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *alpha, double rho)
{
	// xi must be permutated
	double sigma=0;
	switch(param->svm_type)
	{
		case C_SVC:
		{
			int l=prob->l;
			schar *y = new schar[l];
			for(int i=0;i<l;i++)
			{
				if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
			}

			for (int j = 0; j < l; ++j)
			{
				sigma += alpha[j]*prob->y[j]*(kernel_function(prob->x[xi], prob->x[j], *param));
			}
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
	return sigma + rho;
}

double calculate_fx(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *alpha, double rho, double *fx)
{
	// xi must be permutated
	double sigma=0;
	switch(param->svm_type)
	{
		case C_SVC:
		{
			int l=prob->l;
			schar *y = new schar[l];
			for(int i=0;i<l;i++)
			{
				if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
			}

			for (int j = 0; j < l; ++j)
			{
				sigma += alpha[j]*prob->y[j]*(kernel_function(prob->x[xi], prob->x[j], *param));
			}
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
	return sigma + rho;
}

void find_index(int l, int begin_A, int end_A, int begin_R, int end_R, int *valid, int *index)
{
	int count = 0;
	for (int i = 0; i < l; ++i)
	{
		if(i>=begin_A && i<end_A)
		{
			continue;
		}

		if(i>=begin_R && i<end_R)
		{
			continue;
		}

		if(valid[i]>0)
		{
			// record the valid index
			index[count++]=i;
		}
	}
}

void calculate_solution(const struct svm_problem *prob, const struct svm_parameter *param, int *index_M, int count_M, int *index_O, int count_O, int *index_I, int count_I, int *index_A, int count_A, int *index_R, int count_R, Qfloat **Q_ij, double *alpha, int *perm, double rho, double* alpha_St)
{
printf(">>>>>>>>>>> get into approximate_solution\n");
	
	int l = prob->l;
	MatrixXd left_origin(l+1, count_A);
	MatrixXd left_T(count_A, l+1);
	MatrixXd left_multiple(count_A, count_A);
	MatrixXd left_multiple_inv(count_A, count_A);
	MatrixXd second_matrix(l+1, 1);
	MatrixXd up_sum(l, 1);
	MatrixXd down_multiple(1, 1);
	MatrixXd Q_n_Sr(l, count_R);
	MatrixXd alpha_Sr(count_R, 1);
	MatrixXd result(count_A, 1);
	MatrixXd y_T(1, count_R);
	MatrixXd Q_alpha(l, 1);

	double  *delta_f = new double[l];


printf("count_ignored_R+count_ignored_A+count_M+count_O+count_I = %d\n", count_R+count_A+count_M+count_O+count_I);
	
	// calculate the first term in equation (8) on the left
	for (int i = 0; i < l; ++i)
	{
		for (int j = 0; j < count_A; ++j)
		{
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
		delta_f[perm[index_O[i]]] = prob->y[perm[index_O[i]]] - calculate_fx(perm[index_O[i]], prob, param, alpha, rho);
	}

	// I
	for (int i = 0; i < count_I; ++i)
	{
		delta_f[perm[index_I[i]]] = prob->y[perm[index_I[i]]] - calculate_fx(perm[index_I[i]], prob, param, alpha, rho);
	}

	// A
	for (int i = 0; i < count_A; ++i)
	{
		delta_f[perm[index_A[i]]] = 0;
	}

	// Q:Sr & y_T
	for (int i = 0; i < count_R; ++i)
	{
		alpha_Sr(i, 0) = alpha[perm[index_R[i]]];
		y_T(0, i) = prob->y[perm[index_R[i]]];
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

	second_matrix(l, 0) = -1*down_multiple(0, 0);

	left_T = left_origin.transpose();


	left_multiple = left_T*left_origin;

	left_multiple_inv = left_multiple.inverse();

	result = left_multiple_inv*left_T*second_matrix;


	for (int i = 0; i < count_A; ++i)
	{
		alpha_St[i] = result(i, 0);
	}

	delete[] delta_f;

printf("<<<<<<<<<<< get out of approximate_solution\n");
}



void calculate_fi(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int begin_A, int end_A, int *perm, double *f_i)
{
	double sigma = 0;
	for (int i = begin_A; i < end_A; ++i)
	{
		f_i[perm[i]] = 0;
	}

	for (int i = end_A; i < prob->l; ++i)
	{
		sigma = 0;
		for (int j = end_A; j < prob->l; ++j)
		{
			// printf("j = %d y = %lf prob->y[perm[j]]>0 ? 1: -1 = %d\n", j, prob->y[perm[j]], prob->y[perm[j]]>0 ? 1: -1);
			sigma += all_alpha[perm[j]]*(prob->y[perm[j]]>0 ? 1: -1)*kernel_function(prob->x[perm[i]], prob->x[perm[j]], *param);
		}

		f_i[perm[i]] = sigma - prob->y[perm[i]];
	}
}

void calculate_gi(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int begin_A, int end_A, int *perm, double *f_i)
{
	double sigma = 0;
	for (int i = begin_A; i < end_A; ++i)
	{
		f_i[perm[i]] = 0;
	}

	int y_i = 0;
	for (int i = end_A; i < prob->l; ++i)
	{
		sigma = 0;
		y_i = prob->y[perm[i]]>0 ? 1: -1;
		for (int j = end_A; j < prob->l; ++j)
		{
			// printf("j = %d y = %lf prob->y[perm[j]]>0 ? 1: -1 = %d\n", j, prob->y[perm[j]], prob->y[perm[j]]>0 ? 1: -1);
			// TODO all K
			sigma += all_alpha[perm[j]]*y_i*(prob->y[perm[j]]>0 ? 1: -1)*kernel_function(prob->x[perm[i]], prob->x[perm[j]], *param);
		}

		f_i[perm[i]] = sigma - 1;
	}
}

void calculate_gi_K(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, Qfloat **all_K, int begin_A, int end_A, int *perm, double *f_i)
{
	double sigma = 0;
	for (int i = begin_A; i < end_A; ++i)
	{
		f_i[perm[i]] = 0;
	}

	int y_i = 0;
	for (int i = end_A; i < prob->l; ++i)
	{
		sigma = 0;
		y_i = prob->y[perm[i]]>0 ? 1: -1;
		for (int j = end_A; j < prob->l; ++j)
		{
			// printf("j = %d y = %lf prob->y[perm[j]]>0 ? 1: -1 = %d\n", j, prob->y[perm[j]], prob->y[perm[j]]>0 ? 1: -1);
			// TODO all K
			// sigma += all_alpha[perm[j]]*y_i*(prob->y[perm[j]]>0 ? 1: -1)*kernel_function(prob->x[perm[i]], prob->x[perm[j]], *param);
			sigma += all_alpha[perm[j]]*y_i*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[i]][perm[j]];
		}

		f_i[perm[i]] = sigma - 1;
	}
}

void recalculate_gi(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, Qfloat **all_K, int begin, int end, int *perm, double *g_i)
{
	double sigma = 0;
	int i = 0;
	int k = 0;
	// int d = 0;
	// for (d = 0; d < begin; ++d)
	// {
	// 	g_i[d] = -1;
	// }

	// for (d = end; d < prob->l; ++d)
	// {
	// 	g_i[d-end+begin] = -1;
	// }

	for (i = 0; i < begin; ++i)
	{
		sigma = 0;
		for (int j = 0; j < begin; ++j)
		{
			sigma += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[i]][perm[j]];
			// g_i[j] += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[j]][perm[i]];

			// Q_g[k][j] = (Qfloat)(prob->y[perm[j]]>0? 1 : -1)*(Qfloat)(prob->y[perm[j]]>0? 1 : -1)*all_K[perm[j]][perm[j]];
			// Q_g[k][j] = prob->y[perm[j]]*prob->y[perm[j]]*all_K[perm[j]][perm[j]];
		}

		for (int j = end; j < prob->l; ++j)
		{
			sigma += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[i]][perm[j]];
			// g_i[j-end+begin] += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[j]][perm[i]];

			// Q_g[k][j-end] = (Qfloat)(prob->y[perm[j]]>0? 1 : -1)*(Qfloat)(prob->y[perm[j]]>0? 1 : -1)*all_K[perm[j]][perm[j]];
			// Q_g[k][j] = prob->y[perm[j]]*prob->y[perm[j]]*all_K[perm[j]][perm[j]];
		}

		// g_i[perm[i]] = sigma - 1;
		// g_i[perm[k]] = sigma - 1;
		g_i[k] = sigma - 1;
		k++;
	}

	// for (i = begin; i < end; ++i)
	// {
	// 	g_i[perm[i]] = 0;
	// }

	for (i = end; i < prob->l; ++i)
	{
		sigma = 0;
		for (int j = 0; j < begin; ++j)
		{
			sigma += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[i]][perm[j]];
			// g_i[j] += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[j]][perm[i]];
			// Q_g[k][j] = (Qfloat)(prob->y[perm[j]]>0? 1 : -1)*(Qfloat)(prob->y[perm[j]]>0? 1 : -1)*all_K[perm[j]][perm[j]];
			// Q_g[k][j] = prob->y[perm[j]]*prob->y[perm[j]]*all_K[perm[j]][perm[j]];
		}

		for (int j = end; j < prob->l; ++j)
		{
			sigma += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[i]][perm[j]];
			// g_i[j-end+begin] += all_alpha[perm[j]]*(prob->y[perm[i]]>0? 1 : -1)*(prob->y[perm[j]]>0 ? 1: -1)*all_K[perm[j]][perm[i]];
			// Q_g[k][j-end] = (Qfloat)(prob->y[perm[j]]>0? 1 : -1)*(Qfloat)(prob->y[perm[j]]>0? 1 : -1)*all_K[perm[j]][perm[j]];
			// Q_g[k][j] = prob->y[perm[j]]*prob->y[perm[j]]*all_K[perm[j]][perm[j]];
		}
		// g_i[perm[i]] = sigma - 1;
		// g_i[perm[k]] = sigma - 1;
		g_i[k] = sigma - 1;
		k++;
	}
}

// // find the index
// int find_St_index(const struct svm_problem *prob, const struct svm_parameter *param, int Sr_i, int end_A, int count_A, int* index_A, int* valid_A, double *all_alpha, double *f_i, int *perm, double rho)
// {

// 	int min = INT_MAX;
// 	int v_u = 0;
// 	int v_l = 0;
// 	int tmp_i = -1;
// 	int result = -1;
// 	double compare_fi = 0;

// 	// int Gmax_idx = -1;
// 	// int Gmin_idx = -1;
// 	for (int i = 0; i < count_A; ++i)
// 	{
// 		v_u = 0;
// 		v_l = 0;
// 		if(valid_A[i] == 0)
// 		{
// 			continue;
// 		}

// 		// find the min in X_upper, |bu| = |Gmax|
// 		for(int t=end_A;t<prob->l;t++)
// 			if(prob->y[perm[t]]>0)	
// 			{
// 				if(!(all_alpha[perm[t]]>param->C-tol_equal1 && all_alpha[perm[t]]<param->C+tol_equal1))
// 				{
// 					compare_fi = f_i[perm[t]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[t]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[i]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[t]], prob->x[perm[index_A[i]]], *param);
// 					if(compare_fi < rho)
// 					{
// 						v_u++;
// 					}
// 				}
// 			}
// 			else
// 			{ //y[perm[t]]==-1
// 				if(!(all_alpha[perm[t]]>0-tol_equal1 && all_alpha[perm[t]]<0+tol_equal1))
// 				{
// 					compare_fi = -f_i[perm[t]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[t]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[i]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[t]], prob->x[perm[index_A[i]]], *param);
// 					if(compare_fi < rho)
// 					{
// 						v_u++;
// 					}
// 				}
// 			}

// 		// if(v_u>0)
// 		// {

// 		// }

// 		// find the maxima in X_lower, |bl| = |Gmax2|
// 		for(int j=end_A;j<prob->l;j++)
// 		{
// 			if(prob->y[perm[j]]>0)
// 			{
// 				if (!(all_alpha[perm[j]]>0-tol_equal1 && all_alpha[perm[j]]<0+tol_equal1))
// 				{

// 					compare_fi = f_i[perm[j]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[j]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[i]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[j]], prob->x[perm[index_A[i]]], *param);
// 					if(compare_fi > rho)
// 					{
// 						v_l++;
// 					}
					
// 				}
// 			}
// 			else
// 			{
// 				if (!(all_alpha[perm[j]]>param->C-tol_equal1 && all_alpha[perm[j]]<param->C+tol_equal1))
// 				{
// 					compare_fi = -f_i[perm[j]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[j]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[i]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[j]], prob->x[perm[index_A[i]]], *param);
// 					if(compare_fi < rho)
// 					{
// 						v_l++;
// 					}
// 				}
// 			}
// 		}
// 		// printf("i = %d v_u + v_l = %d min = %d\n", i, v_u + v_l, min);
// 		if(v_u + v_l < min)
// 		{
// 			tmp_i = i;
// 			min = v_u + v_l;
// 		}
// 	}
// 	if(tmp_i>=0)
// 	{
// 		valid_A[tmp_i] = 0;
// 		result = index_A[tmp_i];
// 	}
	

// 	printf("result = %d\n", result);
// 	return result;
// }



// find the index
int find_St_index(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat **all_K, double *all_alpha, int Sr_i, double *f_i, int *index_A, int count_A, int * valid_A, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm, double rho)
{
	// printf(">>>>>>>>>>>>> get into find_St_index\n");
	int min_violation = INT_MAX;
	int result = -1;
	int min_index = -1;
	// int upper_v = 0;
	// int lower_v = 0;
	int count_loop = 0;

	int violations = 0;
	double compare_fi = 0;
	for (int a = 0; a < count_A; ++a)
	{
		if(valid_A[a]==0)
		{
			continue;
		}
		count_loop++;


		violations = 0;
		compare_fi = 0;
		for (int i = 0; i < count_X1; ++i)
		{
			// printf("i = %d index_X1[i] = %d perm[index_X1[i]] = %d\n", i, index_X1[i], perm[index_X1[i]]);
			// compare_fi = prob->y[perm[index_X1[i]]]*f_i[perm[index_X1[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X1[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X1[i]]], prob->x[perm[index_A[a]]], *param);
			compare_fi = prob->y[perm[index_X1[i]]]*f_i[perm[index_X1[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X1[i]]][perm[Sr_i]] + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X1[i]]][perm[index_A[a]]];
			if(compare_fi<rho)
			{
				// printf("i = %d index_X1[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X1[i], perm[index_X1[i]], compare_fi, violations);
				violations++;
			}
		}

		// violations *= 2;

		for (int i = 0; i < count_X2; ++i)
		{
			// printf("i = %d index_X2[i] = %d perm[index_X2[i]] = %d\n", i, index_X2[i], perm[index_X2[i]]);
			// compare_fi = prob->y[perm[index_X2[i]]]*f_i[perm[index_X2[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X2[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X2[i]]], prob->x[perm[index_A[a]]], *param);
			compare_fi = prob->y[perm[index_X2[i]]]*f_i[perm[index_X2[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X2[i]]][perm[Sr_i]] + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X2[i]]][perm[index_A[a]]];
			if(compare_fi<rho)
			{
				// printf("i = %d index_X2[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X2[i], perm[index_X2[i]], compare_fi, violations);
				violations++;
			}
		}

		for (int i = 0; i < count_X3; ++i)
		{
			// printf("i = %d index_X3[i] = %d perm[index_X3[i]] = %d\n", i, index_X3[i], perm[index_X3[i]]);
			// compare_fi = prob->y[perm[index_X3[i]]]*f_i[perm[index_X3[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X3[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X3[i]]], prob->x[perm[index_A[a]]], *param);
			compare_fi = prob->y[perm[index_X3[i]]]*f_i[perm[index_X3[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X3[i]]][perm[Sr_i]] + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X3[i]]][perm[index_A[a]]];
			if(compare_fi<rho)
			{
				// printf("i = %d index_X3[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X3[i], perm[index_X3[i]], compare_fi, violations);
				violations++;
			}
		}

		for (int i = 0; i < count_X4; ++i)
		{
			// printf("i = %d index_X4[i] = %d perm[index_X4[i]] = %d\n", i, index_X4[i], perm[index_X4[i]]);
			// compare_fi = prob->y[perm[index_X4[i]]]*f_i[perm[index_X4[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X4[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X4[i]]], prob->x[perm[index_A[a]]], *param);
			compare_fi = prob->y[perm[index_X4[i]]]*f_i[perm[index_X4[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X4[i]]][perm[Sr_i]] + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X4[i]]][perm[index_A[a]]];
			if(compare_fi>rho)
			{
				// printf("i = %d index_X4[i] = %d perm[index_X4[i]] = %d compare_fi = %lf violations = %d\n", i, index_X4[i], perm[index_X4[i]], compare_fi, violations);
				violations++;
			}
		}

		for (int i = 0; i < count_X5; ++i)
		{
			// printf("i = %d index_X5[i] = %d perm[index_X5[i]] = %d\n", i, index_X5[i], perm[index_X5[i]]);
			// compare_fi = prob->y[perm[index_X5[i]]]*f_i[perm[index_X5[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X5[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X5[i]]], prob->x[perm[index_A[a]]], *param);
			compare_fi = prob->y[perm[index_X5[i]]]*f_i[perm[index_X5[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X5[i]]][perm[Sr_i]] + prob->y[perm[index_A[a]]]*all_alpha[perm[Sr_i]]*all_K[perm[index_X5[i]]][perm[index_A[a]]];
			if(compare_fi>rho)
			{
				// printf("i = %d index_X5[i] = %d perm[index_X5[i]] = %d compare_fi = %lf violations = %d\n", i, index_X5[i], perm[index_X5[i]], compare_fi, violations);
				violations++;
			}
		}

		// printf("upper_v = %d lower_v = %d upper_v+lower_v = %d\n", upper_v, lower_v, upper_v+lower_v);

		if( min_violation > violations)
		{
			min_index = a;
			min_violation = violations;
		}
	}

	if(min_index >= 0)
	{
		valid_A[min_index] = 0;
		result = index_A[min_index];
		// printf("result = %d\n", result);
	}

	// printf("count_loop = %d\n", count_loop);
	// printf("<<<<<<<<<<<<< get out of find_St_index\n\n");
	// printf("result = %d\n", result);
	return result;
}

void init_alpha_t(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat **all_K, double *all_alpha,
				  int *index_R,  int count_R, int *valid_0, double *f_i, int *index_A, int count_A, int *valid_A,
				  int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *index_X4, int count_X4,
				  int *index_X5, int count_X5, int *perm, double rho, double* alpha_t)
{
	int min_violation = INT_MAX;
	int result = -1;
	int min_index = -1;

	int violations = 0;
	double compare_fi = 0;
	int tmp_index = -1;
	int tmp_perm_sr = -1;
	int tmp_perm_ia = -1;
	int tmp_swap_a = -1;
	int fixed_count_A = count_A;
	double tmp_sr_ya = 0;
	double tmp_st_ya = 0;
	bool skip = false;
	int negtive = 0;

	printf("R=%d, A=%d, X1=%d, X2=%d, X3=%d, X4=%d, X5=%d\n", count_R, count_A, count_X1, count_X2, count_X3, count_X4, count_X5);

	for (int sri = 0; sri < count_R; ++sri)
	{
		if(valid_0[sri]==1)
		{
			// record_R++;
			continue;
		}

		min_violation = INT_MAX;
		result = -1;
		min_index = -1;

		violations = 0;
		compare_fi = 0;
		tmp_swap_a = -1;

		int count_loop = 0;
		// count_A = fixed_count_A;
		for (int a = 0; a < count_A; ++a)
		{
			// if(valid_A[a]==0)
			// {
			// 	continue;
			// }
			// printf("valid_A: %d\n", a);

			violations = 0;
			compare_fi = 0;
			
			tmp_perm_sr = perm[index_R[sri]];
			tmp_perm_ia = perm[index_A[a]];


			if(!(prob->y[tmp_perm_sr] >= prob->y[tmp_perm_ia] - tol_equal1 && prob->y[tmp_perm_sr] <= prob->y[tmp_perm_ia] + tol_equal1))
			{
				// printf("prob->y[tmp_perm_sr] = %lf prob->y[tmp_perm_ia] = %lf\n", prob->y[tmp_perm_sr], prob->y[tmp_perm_ia]);
				continue;
			}

			tmp_sr_ya = prob->y[tmp_perm_sr]*all_alpha[tmp_perm_sr];
			tmp_st_ya = prob->y[tmp_perm_ia]*all_alpha[tmp_perm_sr];

			skip = false;
			// for (int i = 0; i < count_X1; ++i)
			// {
				
			// 	tmp_index = perm[index_X1[i]];

			// 	compare_fi = prob->y[tmp_index]*f_i[tmp_index] - tmp_sr_ya*all_K[tmp_index][tmp_perm_sr] + tmp_st_ya*all_K[tmp_index][tmp_perm_ia];
				
			// 	if(compare_fi<rho)
			// 	{
			// 		violations++;
			// 		if(violations>min_violation)
			// 		{
			// 			skip = true;
			// 			break;
			// 		}
			// 	}
			// }

			// if(skip)
			// {
			// 	continue;
			// }
			// violations *= 2;

			for (int i = 0; i < count_X2; ++i)
			{
				
				tmp_index = perm[index_X2[i]];

				compare_fi = prob->y[tmp_index]*f_i[tmp_index] - tmp_sr_ya*all_K[tmp_index][tmp_perm_sr] + tmp_st_ya*all_K[tmp_index][tmp_perm_ia];
				if(compare_fi<rho)
				{
					
					violations++;
					if(violations>min_violation)
					{
						skip = true;
						break;
					}
				}
			}

			if(skip)
			{
				continue;
			}

			for (int i = 0; i < count_X3; ++i)
			{
				
				tmp_index = perm[index_X3[i]];

				compare_fi = prob->y[tmp_index]*f_i[tmp_index] - tmp_sr_ya*all_K[tmp_index][tmp_perm_sr] + tmp_st_ya*all_K[tmp_index][tmp_perm_ia];
				if(compare_fi<rho)
				{
					violations++;
					if(violations>min_violation)
					{
						skip = true;
						break;
					}
				}
			}

			if(skip)
			{
				continue;
			}

			for (int i = 0; i < count_X4; ++i)
			{
				
				tmp_index = perm[index_X4[i]];

				compare_fi = prob->y[tmp_index]*f_i[tmp_index] - tmp_sr_ya*all_K[tmp_index][tmp_perm_sr] + tmp_st_ya*all_K[tmp_index][tmp_perm_ia];
				if(compare_fi>rho)
				{
					violations++;
					if(violations>min_violation)
					{
						skip = true;
						break;
					}
				}
			}

			if(skip)
			{
				continue;
			}

			for (int i = 0; i < count_X5; ++i)
			{
				tmp_index = perm[index_X5[i]];

				compare_fi = prob->y[tmp_index]*f_i[tmp_index] - tmp_sr_ya*all_K[tmp_index][tmp_perm_sr] + tmp_st_ya*all_K[tmp_index][tmp_perm_ia];
				if(compare_fi>rho)
				{
					violations++;
					if(violations>min_violation)
					{
						skip = true;
						break;
					}
				}
			}

			if(skip)
			{
				continue;
			}

			// printf("min_violation vs violations: min_violation = %d violations = %d\n", min_violation, violations);

			if( min_violation >= violations)
			{
				min_index = a;
				// printf("choose a = %d index_A[a] = %d count_loop = %d\n", a, index_A[a], count_loop);
				min_violation = violations;
			}
			count_loop++;
		}

		// printf("count_loop = %d\n", count_loop);
		// printf("all data\n");
		// for (int i = 0; i < fixed_count_A; ++i)
		// {
		// 	printf("i = %d index_A[i] = %d\n", i, index_A[i]);
		// }
		
		// printf("min_index = %d \n", min_index);
		if(min_index >= 0)
		{
			// valid_A[min_index] = 0;
			result = index_A[min_index];
			// result = min_index;
			// printf(" result = %d\n", result);
			if(min_index == count_A-1)
			{
				// printf("min_index == count_A-1\n");
				count_A--;
			}
			else{
				// printf("else\n");
				tmp_swap_a = index_A[min_index];
				index_A[min_index] = index_A[count_A-1];
				index_A[count_A-1] = tmp_swap_a;
				count_A--;
			}
			
			// printf("result = %d\n", result);
		}
		else
		{
			negtive++;
		}

		// printf("\n");
		// printf("after ajusting\n");
		// for (int i = 0; i < count_A; ++i)
		// {
		// 	printf("i = %d index_A[i] = %d\n", i, index_A[i]);
		// }
		// printf("\n");

		
		if(result >= 0)
		{
			// record_R++;
			alpha_t[result] = all_alpha[perm[index_R[sri]]];
		}
	}
	// printf("negtive = %d count_R-fixed_count_A = %d\n", negtive, count_R-fixed_count_A);
	// reconstruct index_A
	for (int i = 0; i < fixed_count_A; ++i)
	{
		index_A[i] = i;
	}

	// ================
	// int index_N = new int[negtive];
	// int c_0 = 0;
	// int c_c = 0;
	// double average = 0;

	if(negtive>0)
	{
		return;
	}

	double sum_a = 0;
	for (int i = 0; i < fixed_count_A; ++i)
	{		
		// if(alpha_t[i]<0||(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001))
		// {
		// 	alpha_t[i] = 0;
		// 	// c_0++;
		// }
		// else if(alpha_t[i]>param->C||(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001))
		// {
		// 	alpha_t[i] = param->C;
		// 	// c_c++;
		// }
		sum_a += prob->y[perm[index_A[i]]]*alpha_t[i];
	}

	double sum_r = 0;
	for (int i = 0; i < count_R; ++i)
	{
		sum_r += prob->y[perm[index_R[i]]]*all_alpha[perm[index_R[i]]];
	}

	if(sum_r>sum_a-tol_equal1 && sum_r<sum_a+tol_equal1)
	{
		return;
	}
	else if(sum_r>sum_a)
	{
		printf("get into sum_r>sum_a\n");
		for (int i = 0; i < fixed_count_A; ++i)
		{
			if(prob->y[perm[index_A[i]]]>0)
			{
				if((!(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001))&&(alpha_t[i]+sum_r-sum_a<=param->C))
				{
					alpha_t[i] += sum_r-sum_a;
					break;
				}
			}
			else
			{
				if((!(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001))&&(alpha_t[i]-(sum_r-sum_a)>=0))
				{
					alpha_t[i] -= sum_r-sum_a;
					break;
				}
			}
			
		}
		// average = (sum_r-sum_a)/negtive;
		// for (int i = 0; i < fixed_count_A; ++i)
		// {		
		// 	if((alpha_t[i]<0||(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001)))
		// 	{
		// 		if(prob->y[perm[index_A[i]]]>0)
		// 		{
		// 			alpha_t[i] = sum_r-sum_a;
		// 			break;
		// 		}
				
				
		// 		// c_0++;
		// 	}
		// 	else if(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001)
		// 	{
		// 		if(prob->y[perm[index_A[i]]]<0)
		// 		{
		// 			alpha_t[i] -= sum_r-sum_a;
		// 			break;
		// 		}
		// 	}
		// }
	}
	else if(sum_r<sum_a)
	{
		printf("get into sum_r<sum_a\n");
		for (int i = 0; i < fixed_count_A; ++i)
		{
			if(prob->y[perm[index_A[i]]]>0)
			{
				if((!(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001))&&(alpha_t[i]-(sum_a-sum_r)>=0))
				{
					alpha_t[i] -= sum_a-sum_r;
					break;
				}
			}
			else
			{
				if((!(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001))&&(alpha_t[i]+sum_a-sum_r<=param->C))
				{
					alpha_t[i] += sum_a-sum_r;
					break;
				}
			}

		}
		// for (int i = 0; i < fixed_count_A; ++i)
		// {		
		// 	if((alpha_t[i]<0||(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001)))
		// 	{
		// 		if(prob->y[perm[index_A[i]]]<0)
		// 		{
		// 			alpha_t[i] = sum_a-sum_r;
		// 			break;
		// 		}
				
				
		// 		// c_0++;
		// 	}
		// 	else if(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001)
		// 	{
		// 		if(prob->y[perm[index_A[i]]]>0)
		// 		{
		// 			alpha_t[i] -= sum_a-sum_r;
		// 			break;
		// 		}
		// 	}
		// }
	}
	printf("sum_a = %lf sum_r = %lf\n", sum_a, sum_r);
	
	sum_a = 0;
	for (int i = 0; i < fixed_count_A; ++i)
	{		
		// if(alpha_t[i]<0||(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001))
		// {
		// 	alpha_t[i] = 0;
		// 	// c_0++;
		// }
		// else if(alpha_t[i]>param->C||(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001))
		// {
		// 	alpha_t[i] = param->C;
		// 	// c_c++;
		// }
		sum_a += prob->y[perm[index_A[i]]]*alpha_t[i];
	}

	sum_r = 0;
	for (int i = 0; i < count_R; ++i)
	{
		sum_r += prob->y[perm[index_R[i]]]*all_alpha[perm[index_R[i]]];
	}

	printf("balance two sum: sum_a = %lf sum_r = %lf sum_a-sum_r = %lf\n", sum_a, sum_r, sum_a-sum_r);


}

int upper_violation(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int Sr_i, int St_i, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm, double rho)
{
	// printf("	>>>>>>>>>>>> get into upper_violationp\n");
	int violations = 0;
	double compare_fi = 0;
	for (int i = 0; i < count_X1; ++i)
	{
		// printf("i = %d index_X1[i] = %d perm[index_X1[i]] = %d\n", i, index_X1[i], perm[index_X1[i]]);
		compare_fi = f_i[perm[index_X1[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X1[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[St_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X1[i]]], prob->x[perm[St_i]], *param);
		if(compare_fi<rho)
		{
			printf("i = %d index_X1[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X1[i], perm[index_X1[i]], compare_fi, violations);
			violations++;
		}
	}

	for (int i = 0; i < count_X2; ++i)
	{
		// printf("i = %d index_X2[i] = %d perm[index_X2[i]] = %d\n", i, index_X2[i], perm[index_X2[i]]);
		compare_fi = f_i[perm[index_X2[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X2[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[St_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X2[i]]], prob->x[perm[St_i]], *param);
		if(compare_fi<rho)
		{
			printf("i = %d index_X2[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X2[i], perm[index_X2[i]], compare_fi, violations);
			violations++;
		}
	}

	for (int i = 0; i < count_X3; ++i)
	{
		// printf("i = %d index_X3[i] = %d perm[index_X3[i]] = %d\n", i, index_X3[i], perm[index_X3[i]]);
		compare_fi = f_i[perm[index_X3[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X3[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[St_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X3[i]]], prob->x[perm[St_i]], *param);
		if(compare_fi<rho)
		{
			printf("i = %d index_X3[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X3[i], perm[index_X3[i]], compare_fi, violations);
			violations++;
		}
	}

	// printf("count_X1 + count_X2 + count_X3 = %d\n", count_X1 + count_X2 + count_X3);
	// printf("	<<<<<<<<<<<<< get out of upper_violationp\n");
	return violations;
}

int lower_violation(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int Sr_i, int St_i, double *f_i, int *index_X1, int count_X1, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm, double rho)
{
	// printf("	>>>>>>>>>>>> get into lower_violation\n");
	int violations = 0;
	double compare_fi = 0;
	for (int i = 0; i < count_X1; ++i)
	{
		// printf("i = %d index_X1[i] = %d perm[index_X1[i]] = %d\n", i, index_X1[i], perm[index_X1[i]]);
		compare_fi = f_i[perm[index_X1[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X1[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[St_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X1[i]]], prob->x[perm[St_i]], *param);
		if(compare_fi>rho)
		{
			printf("i = %d index_X1[i] = %d perm[index_X1[i]] = %d compare_fi = %lf violations = %d\n", i, index_X1[i], perm[index_X1[i]], compare_fi, violations);
			violations++;
		}
	}

	for (int i = 0; i < count_X4; ++i)
	{
		// printf("i = %d index_X4[i] = %d perm[index_X4[i]] = %d\n", i, index_X4[i], perm[index_X4[i]]);
		compare_fi = f_i[perm[index_X4[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X4[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[St_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X4[i]]], prob->x[perm[St_i]], *param);
		if(compare_fi>rho)
		{
			printf("i = %d index_X4[i] = %d perm[index_X4[i]] = %d compare_fi = %lf violations = %d\n", i, index_X4[i], perm[index_X4[i]], compare_fi, violations);
			violations++;
		}
	}

	for (int i = 0; i < count_X5; ++i)
	{
		// printf("i = %d index_X5[i] = %d perm[index_X5[i]] = %d\n", i, index_X5[i], perm[index_X5[i]]);
		compare_fi = f_i[perm[index_X5[i]]] - prob->y[perm[Sr_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X5[i]]], prob->x[perm[Sr_i]], *param) + prob->y[perm[St_i]]*all_alpha[perm[Sr_i]]*kernel_function(prob->x[perm[index_X5[i]]], prob->x[perm[St_i]], *param);
		if(compare_fi>rho)
		{
			printf("i = %d index_X5[i] = %d perm[index_X5[i]] = %d compare_fi = %lf violations = %d\n", i, index_X5[i], perm[index_X5[i]], compare_fi, violations);
			violations++;
		}
	}

	// printf("count_X1 + count_X4 + count_X5 = %d\n", count_X1 + count_X4 + count_X5);
	// printf("count_X4 + count_X5 = %d\n", count_X4 + count_X5);
	// printf("	<<<<<<<<<<<<< get out of lower_violation\n");
	return violations;
}

// calculate the minimal fi when xi in X_upper
// double calculate_bu(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm)
// {
// 	double min_b = DBL_MAX;
// 	bool changed = false;
// 	for (int i = 0; i < count_X1; ++i)
// 	{
// 		if(prob->y[perm[index_X1[i]]]<0)
// 		{
// 			continue;
// 		}

// 		if(min_b>-f_i[perm[index_X1[i]]])
// 		{
// 			min_b = -f_i[perm[index_X1[i]]];
// 			changed = true;
// 		}
// 	}

// 	for (int i = 0; i < count_X2; ++i)
// 	{
// 		if(min_b>-f_i[perm[index_X2[i]]])
// 		{
// 			min_b = -f_i[perm[index_X2[i]]];
// 			changed = true;
// 		}
// 	}

// 	for (int i = 0; i < count_X3; ++i)
// 	{
// 		if(min_b>-f_i[perm[index_X3[i]]])
// 		{
// 			min_b = -f_i[perm[index_X3[i]]];
// 			changed = true;
// 		}
// 	}

// 	if(!changed)
// 	{
// 		min_b = 0;
// 	}

// 	return min_b;
// }

// calculate the minimal fi when xi in X_upper
double calculate_bu(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm)
{
	double min_b = DBL_MAX;
	bool changed = false;
	for (int i = 0; i < count_X1; ++i)
	{
		// if(prob->y[perm[index_X1[i]]]<0)
		// {
		// 	continue;
		// }

		if(min_b>f_i[perm[index_X1[i]]])
		{
			min_b = f_i[perm[index_X1[i]]];
			changed = true;
		}
	}

	for (int i = 0; i < count_X2; ++i)
	{
		if(min_b>f_i[perm[index_X2[i]]])
		{
			min_b = f_i[perm[index_X2[i]]];
			changed = true;
		}
	}

	for (int i = 0; i < count_X3; ++i)
	{
		if(min_b>f_i[perm[index_X3[i]]])
		{
			min_b = f_i[perm[index_X3[i]]];
			changed = true;
		}
	}

	if(!changed)
	{
		min_b = 0;
	}

	return min_b;
}


// calculate the minimal fi when xi in X_upper
double calculate_bu_max(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm)
{
	double max_b = -DBL_MAX;
	bool changed = false;
	for (int i = 0; i < count_X1; ++i)
	{
		// if(prob->y[perm[index_X1[i]]]<0)
		// {
		// 	continue;
		// }

		if(max_b<f_i[perm[index_X1[i]]])
		{
			max_b = f_i[perm[index_X1[i]]];
			changed = true;
		}
	}

	for (int i = 0; i < count_X2; ++i)
	{
		if(max_b<f_i[perm[index_X2[i]]])
		{
			max_b = f_i[perm[index_X2[i]]];
			changed = true;
		}
	}

	for (int i = 0; i < count_X3; ++i)
	{
		if(max_b<f_i[perm[index_X3[i]]])
		{
			max_b = f_i[perm[index_X3[i]]];
			changed = true;
		}
	}

	if(!changed)
	{
		max_b = 0;
	}

	return max_b;
}


double calculate_bl(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm)
{
	double max_b = -DBL_MAX;
	bool changed = false;
	for (int i = 0; i < count_X1; ++i)
	{
		if(prob->y[perm[index_X1[i]]]>0)
		{
			continue;
		}
		if(max_b<f_i[perm[index_X1[i]]])
		{
			max_b = f_i[perm[index_X1[i]]];
			changed = true;
		}
	}

	for (int i = 0; i < count_X4; ++i)
	{
		if(max_b<f_i[perm[index_X4[i]]])
		{
			max_b = f_i[perm[index_X4[i]]];
			changed = true;
		}
	}

	for (int i = 0; i < count_X5; ++i)
	{
		if(max_b<f_i[perm[index_X5[i]]])
		{
			max_b = f_i[perm[index_X5[i]]];
			changed = true;
		}
	}

	if(!changed)
	{
		max_b = 0;
	}

	return max_b;
}

void my_select_working_set(const struct svm_problem *prob, const struct svm_parameter *param, int end_A, double *all_alpha, double *f_i, int *perm)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -DBL_MAX;
	double Gmax2 = -DBL_MAX;
	// int Gmax_idx = -1;
	// int Gmin_idx = -1;

	// find the min in X_upper, |bu| = |Gmax|
	for(int t=end_A;t<prob->l;t++)
		if(prob->y[perm[t]]>0)	
		{
			if(!(all_alpha[perm[t]]>param->C-tol_equal1 && all_alpha[perm[t]]<param->C+tol_equal1))
				if(-f_i[perm[t]] >= Gmax)
				{
					Gmax = -f_i[perm[t]];
					// Gmax_idx = t;
				}
		}
		else
		{ //y[perm[t]]==-1
			if(!(all_alpha[perm[t]]>0-tol_equal1 && all_alpha[perm[t]]<0+tol_equal1))
				if(f_i[perm[t]] >= Gmax)
				{
					Gmax = f_i[perm[t]];
					// Gmax_idx = t;
				}
		}


	// find the maxima in X_lower, |bl| = |Gmax2|
	for(int j=end_A;j<prob->l;j++)
	{
		if(prob->y[perm[j]]>0)
		{
			if (!(all_alpha[perm[j]]>0-tol_equal1 && all_alpha[perm[j]]<0+tol_equal1))
			{
				if (f_i[perm[j]] >= Gmax2)
				{	
					Gmax2 = f_i[perm[j]];
					// Gmin_idx=j;
				}
				
			}
		}
		else
		{
			if (!(all_alpha[perm[j]]>param->C-tol_equal1 && all_alpha[perm[j]]<param->C+tol_equal1))
			{
				if (-f_i[perm[j]] >= Gmax2)
				{
					Gmax2 = -f_i[perm[j]];
					// Gmin_idx=j;
				}
			}
		}
	}

	printf("in my bu = -Gmax = %lf bl = Gmax2 = %lf difference = %lf\n", -Gmax, Gmax2, -Gmax-Gmax2);

	// if(Gmax+Gmax2 < param->eps || Gmin_idx == -1)
	// {
	// 	printf("in my bu = -Gmax = %lf bl = Gmax2 = %lf\n", -Gmax, Gmax2);
	// }

	// out_i = Gmax_idx;
	// out_j = Gmin_idx;
}

void adjust_sum_ya(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, double *alpha_t, int *index_A, int count_A, int *index_R, int count_R, int *perm)
{
	int greater = 0;
	int smaller = 0;
	int count_for_average = 0;
	double sum_a = 0;
	// double tmp_a_t = 0;
	for (int i = 0; i < count_A; ++i)
	{
		// method 3
		// tmp_a_t = alpha_t[i];
		
		if(alpha_t[i]<0||(alpha_t[i]>-0.0000001&&alpha_t[i]<0.0000001))
		{
			alpha_t[i] = 0;
			count_for_average++;
		}
		else if(alpha_t[i]>param->C||(alpha_t[i]>param->C-0.0000001&&alpha_t[i]<param->C+0.0000001))
		{
			alpha_t[i] = param->C;
			count_for_average++;
		}
		sum_a += prob->y[perm[index_A[i]]]*alpha_t[i];
	}

	double sum_r = 0;
	for (int i = 0; i < count_R; ++i)
	{
		sum_r += prob->y[perm[index_R[i]]]*all_alpha[perm[index_R[i]]];
	}

	// check if these two sums are equal
	printf("Compare sigma_St(A) vs. sigma_Sr(R): %lf vs. %lf\n", sum_a, sum_r);

	if(sum_a>sum_r)
	{
		double average = (double)(sum_a-sum_r)/(count_A-count_for_average);
		printf("sum_a>sum_r, sum_a-sum_r = %lf greater = %d average = %lf count_A = %d count_for_average =%d\n", sum_a-sum_r, greater, average, count_A, count_for_average);
		int count_check = 0;
		for (int m = 0; m < count_A; ++m)
		{

			if(((alpha_t[m]>0-tol_equal1)&&(alpha_t[m]<0+tol_equal1)))
			{
				all_alpha[perm[index_A[m]]] = 0;
				count_check++;
				continue;
			}

			if((alpha_t[m]>param->C-tol_equal1)&&(alpha_t[m]<param->C+tol_equal1))
			{
				all_alpha[perm[index_A[m]]] = param->C;
				count_check++;
				continue;
			}

			if(prob->y[perm[index_A[m]]]<0)
			{
				all_alpha[perm[index_A[m]]] = alpha_t[m] + average;
				greater++;
			}
			else
			{
				all_alpha[perm[index_A[m]]] = alpha_t[m] - average;
				greater++;
			}
			
		}
		printf("if greater is equal to count_check: %d vs. %d sum = %d \n", greater, count_check, greater+count_check );
	}
	else
	{
		// sum_a<sum_r
		double average = (double)(sum_r-sum_a)/(count_A-count_for_average);
		printf("sum_a<sum_r, sum_r-sum_a = %lf average = %lf count_A = %d count_for_average = %d\n", sum_r-sum_a, average, count_A, count_for_average);
		int count_check = 0;
		for (int m = 0; m < count_A; ++m)
		{
			if(((alpha_t[m]>0-tol_equal1)&&(alpha_t[m]<0+tol_equal1)))
			{
				all_alpha[perm[index_A[m]]] = 0;
				count_check++;
				continue;
			}

			if((alpha_t[m]>param->C-tol_equal1)&&(alpha_t[m]<param->C+tol_equal1))
			{
				all_alpha[perm[index_A[m]]] = param->C;
				count_check++;
				continue;
			}

			if(prob->y[perm[index_A[m]]]<0)
			{
				all_alpha[perm[index_A[m]]] = alpha_t[m] - average;	
				smaller++;
			}
			else
			{
				all_alpha[perm[index_A[m]]] = alpha_t[m] + average;
				smaller++;
			}
			// printf("i = %d alpha_t vs. alpha_St: %lf vs. %lf\n", m, alpha_t[m], all_alpha[index_A[m]]);
		}
		printf("if smaller is equal to count_check: %d vs. %d sum = %d \n", smaller, count_check, smaller+count_check);
	}


	// check these two sums are equal or not again, after adjusting
	sum_a = 0;
	for (int i = 0; i < count_A; ++i)
	{
		sum_a += prob->y[perm[index_A[i]]]*all_alpha[perm[index_A[i]]];
	}

	sum_r = 0;
	for (int i = 0; i < count_R; ++i)
	{
		sum_r += prob->y[perm[index_R[i]]]*all_alpha[perm[index_R[i]]];
	}
	printf("Second comparing sigma_St(A) vs. sigma_Sr(R): %lf vs. %lf\n", sum_a, sum_r);
}
