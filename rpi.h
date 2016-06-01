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
void calculate_Kernel(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_K);

// calculate f(x_i)
double calculate_fx(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *alpha, double rho);

double calculate_fx(int xi, const struct svm_problem *prob, const struct svm_parameter *param, double *alpha, double rho, double *fx);

// find index
void find_index(int l, int begin_A, int end_A, int begin_R, int end_R, int *valid, int *index);

void calculate_solution(const struct svm_problem *prob, const struct svm_parameter *param, int *index_M, int count_M, int *index_O, int count_O, int *index_I, int count_I, int *index_A, int count_A, int *index_R, int count_R, Qfloat **Q_ij, double *alpha, int *perm, double rho, double* alpha_St);

// just checking Eigen
void checking();

// calculate array fi
void calculate_fi(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int begin_A, int end_A, int *perm, double *f_i);

void calculate_gi(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int begin_A, int end_A, int *perm, double *f_i);

int find_St_index(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat ** all_K, double *all_alpha, int Sr_i, double *f_i, int *index_A, int count_A, int *valid_A, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm, double rho);
// int find_St_index(const struct svm_problem *prob, const struct svm_parameter *param, int Sr_i, int end_A, int count_A, int* index_A, int* valid_A, double *all_alpha, double *f_i, int *perm, double rho);

int upper_violation(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int Sr_i, int St_i, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm, double rho);

int lower_violation(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, int Sr_i, int St_i, double *f_i, int *index_X1, int count_X1, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm, double rho);


double calculate_bu(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm);
double calculate_bu_max(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *perm);
double calculate_bl(const struct svm_problem *prob, double *f_i, int *index_X1, int count_X1, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm);

void my_select_working_set(const struct svm_problem *prob, const struct svm_parameter *param, int end_A, double *all_alpha, double *f_i, int *perm);

void init_alpha_t(const struct svm_problem *prob, const struct svm_parameter *param, Qfloat **all_K, double *all_alpha, int *index_R,  int count_R, int *valid_0, double *f_i, int *index_A, int count_A, int *valid_A, int *index_X1, int count_X1, int *index_X2, int count_X2, int *index_X3, int count_X3, int *index_X4, int count_X4, int *index_X5, int count_X5, int *perm, double rho, double* alpha_t);

void adjust_sum_ya(const struct svm_problem *prob, const struct svm_parameter *param, double *all_alpha, double *alpha_t, int *index_A, int count_A, int *index_R, int count_R, int *perm);