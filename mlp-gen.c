#include "autodiff.c" // XXX make it so single-header lib

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INDENT "  "

void matnan(int m, int n, struct node *out[m][n]) {
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      out[i][j] = LIT(NAN);
}

void matmul(int m, int n, int o, struct node *lhs[m][n], struct node *rhs[n][o],
            struct node *out[m][o]) {
  for (int k = 0; k < o; k++) {
    for (int i = 0; i < m; i++) {
      out[i][k] = LIT(0.0);
      for (int j = 0; j < n; j++)
        out[i][k] = ADD(out[i][k], MUL(lhs[i][j], rhs[j][k]));
    }
  }
}

void matunop(int m, int n, struct node *lhs[m][n],
             struct node *(*unop)(struct node *lhs), struct node *out[m][n]) {
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      out[i][j] = unop(lhs[i][j]);
}

void matbinop(int m, int n, struct node *lhs[m][n], struct node *rhs[m][n],
              struct node *(*binop)(struct node *lhs, struct node *rhs),
              struct node *out[m][n]) {
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      out[i][j] = binop(lhs[i][j], rhs[i][j]);
}

struct node *mataccum(int m, int n, struct node *mat[m][n],
                      struct node *(*binop)(struct node *lhs, struct node *rhs),
                      struct node *e) {
  struct node *accum = e;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      accum = binop(accum, mat[i][j]);
  return accum;
}

struct node *sigmoid(struct node *x) { return INV(ADD(LIT(1.0), EXP(NEG(x)))); }

// TODO test
struct node *tanh_(struct node *x) {
  struct node *exp_2x = EXP(ADD(x, x));
  return DIV(SUB(exp_2x, LIT(1.0)), ADD(exp_2x, LIT(2.0)));
}

// TODO make other demos? matrix factorization, xor, curve fitting

int main(int argc, char **argv) {
  static struct mlp {
    struct x {
      struct node *x[784];
      struct node *yh[10];
    } x;
    struct w {
      struct node *w1[16][784];
      struct node *w2[16][16];
      struct node *w3[10][16];
      struct node *b1[16];
      struct node *b2[16];
      struct node *b3[10];
    } w;
    struct l {
      struct node *l1[16];
      struct node *l2[16];
      struct node *l3[10];
    } l;
    struct node *c;
  } mlp;

  // XXX UB?
  struct node **params = (struct node **)mlp.w.w1;
  struct node **outs = (struct node **)mlp.l.l3;

  matnan(784, 1, &mlp.x.x);
  matnan(10, 1, &mlp.x.yh);

  matnan(16, 784, mlp.w.w1);
  matnan(16, 1, &mlp.w.b1);
  matmul(16, 784, 1, mlp.w.w1, &mlp.x.x, &mlp.l.l1);
  matbinop(16, 1, &mlp.w.b1, &mlp.l.l1, ADD, &mlp.l.l1);
  matunop(16, 1, &mlp.l.l1, sigmoid, &mlp.l.l1);

  matnan(16, 16, mlp.w.w2);
  matnan(16, 1, &mlp.w.b2);
  matmul(16, 16, 1, mlp.w.w2, &mlp.l.l1, &mlp.l.l2);
  matbinop(16, 1, &mlp.w.b2, &mlp.l.l2, ADD, &mlp.l.l2);
  matunop(16, 1, &mlp.l.l2, sigmoid, &mlp.l.l2);

  matnan(10, 16, mlp.w.w3);
  matnan(10, 1, &mlp.w.b3);
  matmul(10, 16, 1, mlp.w.w3, &mlp.l.l2, &mlp.l.l3);
  matbinop(10, 1, &mlp.w.b3, &mlp.l.l3, ADD, &mlp.l.l3);
  matunop(10, 1, &mlp.l.l3, sigmoid, &mlp.l.l3);

  struct node *mse[10];
  matbinop(10, 1, &mlp.x.yh, &mlp.l.l3, SUB, &mse);
  matbinop(10, 1, &mse, &mse, MUL, &mse);
  mlp.c = DIV(mataccum(10, 1, &mse, ADD, LIT(0.0)), LIT(10.0));

  int visited = 0;
  FILE *fp = fopen(argv[1], "w"); // XXX bounds check
  if (fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);

  fprintf(fp, "#include <math.h>\n\n");

  fprintf(fp, "void predict(double x[], double w[], double y[]) {\n");
  for (int i = 0; i < 784; i++)
    fprintf(fp, INDENT "double t%d = x[%d];\n", mlp.x.x[i]->id, i);
  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    fprintf(fp, INDENT "double t%d = w[%d];\n", params[i]->id, i);
  putc('\n', fp);
  ++visited;
  for (int i = 0; i < 10; i++)
    node_gen(fp, outs[i], visited);
  putc('\n', fp);
  for (int i = 0; i < 10; i++)
    fprintf(fp, INDENT "y[%d] = t%d;\n", i, outs[i]->id);
  fprintf(fp, "}\n\n");

  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    params[i]->grad = LIT(0.0);
  mlp.c->grad = LIT(1.0);
  node_grad(mlp.c, ++visited);

  fprintf(fp, "void backprop(double x[], double w[],"
              "double yh[], double dw[], double *c) {\n");
  for (int i = 0; i < 784; i++)
    fprintf(fp, INDENT "double t%d = x[%d];\n", mlp.x.x[i]->id, i);
  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    fprintf(fp, INDENT "double t%d = w[%d];\n", params[i]->id, i);
  for (int i = 0; i < 10; i++)
    fprintf(fp, INDENT "double t%d = yh[%d];\n", mlp.x.yh[i]->id, i);
  putc('\n', fp);
  node_gen(fp, mlp.c, ++visited);
  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    node_gen(fp, params[i]->grad, visited);
  putc('\n', fp);
  fprintf(fp, INDENT "*c += t%d;\n", mlp.c->id);
  for (int i = 0; i < 16 * 784 + 16 * 16 + 10 * 16 + 16 + 16 + 10; i++)
    fprintf(fp, INDENT "dw[%d] += t%d;\n", i, params[i]->grad->id);
  fprintf(fp, "}\n");

  if (fclose(fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  // node_free(node_ll); // XXX run against asan
}
