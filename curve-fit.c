#include "lib/autodiff.h"
#include "lib/tensor.h"
#include "lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ETA 0.002     // learning rate
#define DEGREE 3      // degree of polynomial (plus one)
#define ITERS 1000000 // number of iterations performed

#define NPOINTS 20
#define POINT_Y(N) N
#define POINT_X(N) -0.06 * N *N + 1.0 * N + 5.0
#define NOISE_X(N) (double)rand() / RAND_MAX - 0.5;
#define NOISE_Y(N) (double)rand() / RAND_MAX - 0.5;

int main(void) {
  srand(time(NULL));

  int visited = 0;

  struct tensor *x = tensor_nans((shape_t){NPOINTS});
  struct tensor *w = tensor_nans((shape_t){DEGREE});
  struct tensor *yh = tensor_lit(x->shape, LIT(0.0));
  TENSOR_FOR(w)
  yh = tensor_binop(ADD, tensor_binop(MUL, yh, x), tensor_lit(yh->shape, node));

  struct tensor *y = tensor_nans(yh->shape);
  struct node *r2 = tensor_r2(y, yh);

  TENSOR_FOR(w) node->grad = LIT(0.0);
  r2->grad = LIT(1.0), node_grad(r2, ++visited);

  TENSOR_FOR(x) node->val = POINT_Y(idx) + NOISE_X(idx);
  TENSOR_FOR(y) node->val = POINT_X(idx) + NOISE_Y(idx);
  TENSOR_FOR(w) node->val = (double)rand() / RAND_MAX - 0.5;
  for (int iter = 0; iter < ITERS; iter++) {
    node_eval(r2, ++visited);
    TENSOR_FOR(w) node_eval(node->grad, visited);
    TENSOR_FOR(w) node->val -= ETA * node->grad->val / shape_size(x->shape);

    if (iter % 1000 == 0)
      printf("iter %d of %d; r2 %f\n", iter, ITERS, r2->val);
  }

#define STRINGIZE_INNER(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_INNER(__VA_ARGS__)

  printf(STRINGIZE(POINT_Y(y)) " = " STRINGIZE(POINT_X(x)) "\n");
  printf("yh = ");
  TENSOR_FOR(w) putchar('(');
  printf("0.0");
  TENSOR_FOR(w) printf(") * x %+f", node->val);
  putchar('\n');

  // XXX doc no free
}
