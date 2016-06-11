#include <math.h>
#include <stdio.h>
// #include <armadillo>
#include <Eigen/Dense>
#include <float.h>

#include "rpi.h"
#include "svm.h"

// using namespace arma;
using namespace Eigen;

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
// test: bean
// printf("the kernel_type is %d\n", param.kernel_type);

	switch(param.kernel_type)
	{
		case LINEAR:
			return Kernel_dot(x,y);
		case POLY:
			{
// printf("get in POLY\n");
				return pow(param.gamma*Kernel_dot(x,y)+param.coef0,param.degree);
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

// void calculate_solution(const struct svm_problem *prob, const struct svm_parameter *param, int *index_M, int count_M, int *index_O, int count_O, int *index_I, int count_I, int *index_A, int count_A, int *index_R, int count_R, Qfloat **Q_ij, double *alpha, int *perm, double rho, double* alpha_St)
// {
// printf(">>>>>>>>>>> get into approximate_solution\n");
	
// 	int l = prob->l;
// 	mat left_origin(l+1, count_A);
// 	mat left_T(count_A, l+1);
// 	mat left_multiple(count_A, count_A);
// 	mat left_multiple_inv(count_A, count_A);
// 	mat second_matrix(l+1, 1);
// 	mat up_sum(l, 1);
// 	mat down_multiple(1, 1);
// 	mat Q_n_Sr(l, count_R);
// 	mat alpha_Sr(count_R, 1);
// 	mat result(count_A, 1);
// 	mat y_T(1, count_R);
// 	mat Q_alpha(l, 1);

// 	double  *delta_f = new double[l];

// // 	// print testing
// // // printf("count_M = %d\n", count_M);
// // // printf("count_O = %d\n", count_O);
// // // printf("count_I = %d\n", count_I);
// // // printf("count_A = %d\n", count_A);
// // // printf("count_R = %d\n", count_R);

// printf("count_ignored_R+count_ignored_A+count_M+count_O+count_I = %d\n", count_R+count_A+count_M+count_O+count_I);
	
// 	// calculate the first term in equation (8) on the left
// 	for (int i = 0; i < l; ++i)
// 	{
// 		for (int j = 0; j < count_A; ++j)
// 		{
// 			left_origin(i, j) = Q_ij[i][perm[index_A[j]]];
// 		}
// 	}


// 	for (int i = 0; i < count_A; ++i)
// 	{
// 		left_origin(l, i) = prob->y[perm[index_A[i]]];
// 	}

// 	// calculate the delta_f in the second term
// 	// M
// 	for (int i = 0; i < count_M; ++i)
// 	{
// 		delta_f[perm[index_M[i]]] = 0;
// 	}

// 	// O
// 	for (int i = 0; i < count_O; ++i)
// 	{
// 		delta_f[perm[index_O[i]]] = prob->y[perm[index_O[i]]] - calculate_fx(perm[index_O[i]], prob, param, alpha, rho);
// 	}

// 	// I
// 	for (int i = 0; i < count_I; ++i)
// 	{
// 		delta_f[perm[index_I[i]]] = prob->y[perm[index_I[i]]] - calculate_fx(perm[index_I[i]], prob, param, alpha, rho);
// 	}

// 	// A
// 	for (int i = 0; i < count_A; ++i)
// 	{
// 		delta_f[perm[index_A[i]]] = 0;
// 	}

// 	// Q:Sr & y_T
// 	for (int i = 0; i < count_R; ++i)
// 	{
// 		alpha_Sr(i, 0) = alpha[perm[index_R[i]]];
// 		y_T(0, i) = prob->y[perm[index_R[i]]];
// 	}

// 	// alpha_Sr
// 	for (int i = 0; i < l; ++i)
// 	{
// 		for (int j = 0; j < count_R; ++j)
// 		{
// 			Q_n_Sr(i, j) = Q_ij[i][perm[index_R[j]]];
// 		}
// 	}

// 	up_sum = Q_n_Sr*alpha_Sr;

// 	for (int i = 0; i < l; ++i)
// 	{
// 		second_matrix(i, 0) = prob->y[i]*delta_f[i] + up_sum(i, 0);
// 	}

// 	down_multiple = y_T * alpha_Sr;

// 	second_matrix(l, 0) = -1*down_multiple(0, 0);

// 	left_T = trans(left_origin);


// 	left_multiple = left_T*left_origin;

// 	left_multiple_inv = inv(left_multiple);

// 	result = left_multiple_inv*left_T*second_matrix;


// 	for (int i = 0; i < count_A; ++i)
// 	{
// 		alpha_St[i] = result(i, 0);
// 	}

// 	delete[] delta_f;

// printf("<<<<<<<<<<< get out of approximate_solution\n");
// }

void calculate_solution(const struct svm_problem *prob, const struct svm_parameter *param, int *index_M, int count_M, 
			int *index_O, int count_O, int *index_I, int count_I, int *index_A, int count_A, int *index_R, int count_R, 
			Qfloat **K_ij, double *alpha, int *perm, double rho, double* alpha_St)
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

// 	// print testing
// // printf("count_M = %d\n", count_M);
// // printf("count_O = %d\n", count_O);
// // printf("count_I = %d\n", count_I);
// // printf("count_A = %d\n", count_A);
// // printf("count_R = %d\n", count_R);

printf("count_ignored_R+count_ignored_A+count_M+count_O+count_I = %d\n", count_R+count_A+count_M+count_O+count_I);
	
	// calculate the first term in equation (8) on the left
	for (int i = 0; i < l; ++i)
	{
		for (int j = 0; j < count_A; ++j)
		{
			left_origin(i, j) = (Qfloat)(prob->y[i]>0?+1:-1)*(prob->y[perm[index_A[j]]]>0?+1:-1)*K_ij[i][perm[index_A[j]]];
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
			Q_n_Sr(i, j) = (Qfloat)(prob->y[i]>0?+1:-1)*(prob->y[perm[index_R[j]]]>0?+1:-1)*K_ij[i][perm[index_R[j]]];
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
		if(isnan(result(i, 0))||result(i, 0)<0)
		{
			alpha_St[i] = 0;
		}
		else if(result(i, 0 )>param->C)
		{
			alpha_St[i] = param->C;
		}
		else{
			alpha_St[i] = result(i, 0);	
		}
	}

	delete[] delta_f;

printf("<<<<<<<<<<< get out of approximate_solution\n");
}


// void checking()
// {
// 	// mat a(1,1);
// 	// a(0,0) = 1;
// 	// mat b(1,1);
// 	// b(0,0) = 1;
// 	// mat c(1,1);
// 	// c = trans(a);
// 	// mat result(1, 1);
// 	// result = a*b;

// 	// printf("a(1,1) = %lf\n", result(0,0));
// 	MatrixXd m(300,2266);

// 	// m(0,0) = 3;
// 	// m(1,0) = 2.5;
// 	// m(0,1) = -1;
// 	// m(1,1) = m(1,0) + m(0,1);

// 	MatrixXd n(2266,300);
// 	for (int i = 0; i < 300; ++i)
// 	{
// 		for (int j = 0; j < 2266; ++j)
// 		{
// 			m(i,j) = i+j;
// 			n(j,i) = 2266+300-i;
// 		}
// 	}
//     // n(0,0) = 3;
//     // n(1,0) = 2.5;
//     // n(0,1) = -1;
//     // n(1,1) = n(1,0) + n(0,1);

//     MatrixXd o = m*n;
//   // std::cout << "Here is the matrix m:\n" << m << std::endl;
//   printf("Here is the matrix m: %lf\n", o(0,0));
//   // std::cout << "Here is the vector v:\n" << v << std::endl;
// }

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

// calculate K_ij
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
					if(j < i)
					{
						all_K[i][j] = all_K[j][i];
					}
					else
					{
						all_K[i][j] = (Qfloat)(kernel_function(prob->x[i], prob->x[j], *param));	
					}
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
