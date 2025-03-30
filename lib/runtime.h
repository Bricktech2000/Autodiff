// runtime for `struct node`. provides implementations for all node types, with
// homogeneous naming, so the homomorphism can be established programmatically
// using the preprocessor

#include <math.h>

#define op_lit(LIT) LIT

#define op_add(LHS, RHS) LHS + RHS
#define op_sub(LHS, RHS) LHS - RHS
#define op_neg(LHS) -LHS

#define op_mul(LHS, RHS) LHS *RHS
#define op_div(LHS, RHS) LHS / RHS
#define op_inv(LHS) 1.0 / LHS

#define op_exp(LHS) exp(LHS)
#define op_log(LHS) log(LHS)
#define op_exp2(LHS) exp2(LHS)
#define op_log2(LHS) log2(LHS)

#define op_pow(LHS, RHS) pow(LHS, RHS)
#define op_sqrt(LHS) sqrt(LHS)
#define op_cbrt(LHS) cbrt(LHS)

#define op_min(LHS, RHS) fmin(LHS, RHS)
#define op_max(LHS, RHS) fmax(LHS, RHS)
#define op_relu(LHS) fmax(LHS, 0.0)
#define op_abs(LHS) fabs(LHS)
