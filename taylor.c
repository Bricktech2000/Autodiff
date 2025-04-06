#include "lib/autodiff.h"
#include <math.h>

#define FUNC(X) node_log(X)          // function to approximate
#define CENTER (LOWER + UPPER) / 2.0 // center of expansion
#define DEGREE 8                     // degree of polynomial (plus one)

#define LOWER 1.0      // closed lower bound for test interval
#define UPPER exp(1.0) // open upper bound for test interval
#define STEPS 256      // number of steps in test interval

struct node *taylor(struct node *f, struct node *x, int N, int *visited) {
  // returns the `N`th taylor polynomial of `f(x)` around the center of
  // expansion `x->val`. consumes `f` and returns a function of `x`

  struct node *x0 = node_sub(x, node_lit(x->val));

  double fact_n = 1.0;   // factorial of `n`
  struct node *df_n = f; // `n`th derivative of `f`
  struct node *p_n =     // `n`th taylor polynomial
      node_lit(N ? node_eval(f, ++*visited), f->val : 0.0);
  struct node *x0_n = node_lit(1.0); // `n`th power of `x - x->val`

  for (int n = 1; n < N; n++) {
    // derivate `df_n` with respect to `x`
    struct node *grad;
    x->grad = node_lit(0.0); // assume `f != x`
    df_n->grad = node_lit(1.0), node_grad(df_n, ++*visited);
    grad = x->grad, x->grad = NULL;

    // garbage collect
    struct node *nodes = NULL, *rest = NULL;
    node_mark(x0, &nodes, 0, ++*visited); // `grad` may not depend on `x0`
    node_mark(grad, &nodes, 0, *visited), node_mark(df_n, &rest, 0, *visited);
    node_zerograd(nodes, *visited), node_free(rest, *visited);

    // build the term
    node_eval(df_n = grad, ++*visited);
    x0_n = node_mul(x0_n, x0), fact_n *= n;
    p_n = node_add(p_n, node_mul(node_lit(df_n->val / fact_n), x0_n));
  }

  struct node *nodes = NULL;
  node_mark(p_n, NULL, 0, ++*visited);
  node_mark(df_n, &nodes, 0, *visited), node_free(nodes, *visited);

  return p_n;
}

int main(void) {
  int visited = 0;
  FILE *fp = stdout;

  struct node *x = node_lit(CENTER), *f = FUNC(x);
  struct node *p_n = taylor(FUNC(x), x, DEGREE, &visited);

  x->val = NAN;
  fprintf(fp, "#include \"runtime.h\"\n");
  fprintf(fp, "double p_n(double x) {\n");
  fprintf(fp, "double t%d = x;\n", x->id);
  node_codegen(fp, "double t%d = ", "t%d", p_n, ++visited);
  fprintf(fp, "return t%d;\n", p_n->id);
  fprintf(fp, "}\n");

  double rmse = 0.0, l_inf = 0.0;
  for (int step = 0; step < STEPS; step++) {
    x->val = LOWER + (double)(UPPER - LOWER) / STEPS * step;
    node_eval(f, ++visited), node_eval(p_n, visited);
    double delta = f->val - p_n->val;
    rmse += delta * delta, l_inf = fmax(l_inf, fabs(delta));
  }
  rmse = sqrt(rmse / STEPS);
  printf("rmse: %f; l_inf: %f\n", rmse, l_inf);

  struct node *nodes = NULL;
  node_mark(f, &nodes, 0, ++visited);
  node_mark(p_n, &nodes, 0, visited), node_free(nodes, visited);
}
