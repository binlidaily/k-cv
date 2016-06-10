#include "svm.h"


// add defination
typedef float Qfloat;

// powi
double pow(double base, int times);

// redefine the dot function for caculating f(x_i) with poly kernel
double Kernel_dot(const svm_node *px, const svm_node *py);

// calculate kernal value
double kernel_function(const svm_node *x, const svm_node *y, const svm_parameter& param);

// calculate Q_ij
void calculate_Q(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_Q);
void recalculate_gi(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, Qfloat **all_K, int begin, int end, int *perm, double *g_i);
// calculate f(x_i)
double calculate_fx(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *alpha, double rho);

double calculate_fx(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *alpha, double rho, double *fx);

// find index
void find_index(int l, int begin_A, int end_A, int begin_R, int end_R, int *valid, int *index);

void calculate_solution(const struct svm_problem *prob, const struct svm_parameter *param, int *index_M, int count_M, int *index_O, int count_O, int *index_I, int count_I, int *index_A, int count_A, int *index_R, int count_R, Qfloat **Q_ij, double *alpha, int *perm, double rho, double* alpha_St);

void checking();


void calculate_Kernel(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_K);