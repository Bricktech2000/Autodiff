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
#define POINT_X(T) T
#define POINT_Y(T) -0.06 * T *T + 1.0 * T + 5.0
#define NOISE_X(T) (double)rand() / RAND_MAX - 0.5
#define NOISE_Y(T) (double)rand() / RAND_MAX - 0.5

int main(void) {
  srand(time(NULL));

  int visited = 0;

  struct tensor *x = tensor_nans((shape_t){NPOINTS});
  struct tensor *w = tensor_nans((shape_t){DEGREE});
  struct tensor *yh = tensor_repeat(x->shape, node_lit(0.0));
  TENSOR_FOR(w)
  yh = tensor_binop(node_add, MOVE tensor_binop(node_mul, MOVE yh, REF x),
                    MOVE tensor_repeat(yh->shape, node));

  struct tensor *y = tensor_nans(yh->shape);
  struct node *r2 = tensor_r2(REF y, REF yh);

  TENSOR_FOR(w) node->grad = node_lit(0.0);
  r2->grad = node_lit(1.0), node_grad(r2, ++visited);

  TENSOR_FOR(x) node->val = POINT_X(idx) + NOISE_X(idx);
  TENSOR_FOR(y) node->val = POINT_Y(idx) + NOISE_Y(idx);
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

  printf("(" STRINGIZE(POINT_X(t)) ", " STRINGIZE(POINT_Y(t)) ")\n");
  printf("(t, ");
  TENSOR_FOR(w) putchar('(');
  printf("0.0");
  TENSOR_FOR(w) printf(") * t %+f", node->val);
  printf(")\n");

#undef STRINGIZE_INNER
#undef STRINGIZE

  r2->next = NULL, node_free(r2, ++visited);
  free(x), free(yh), free(w), free(y);
}
