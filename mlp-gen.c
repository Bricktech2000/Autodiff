#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  struct tensor *l0 = COL_TENSOR(tensor_nans((shape_t){784}));

  struct tensor *b1 = COL_TENSOR(tensor_nans((shape_t){64}));
  struct tensor *w1 = tensor_nans((shape_t){*b1->shape, *l0->shape});
  struct tensor *l1 =
      tensor_unop(RELU, tensor_binop(ADD, b1, tensor_matmul(w1, l0)));

  struct tensor *b2 = COL_TENSOR(tensor_nans((shape_t){32}));
  struct tensor *w2 = tensor_nans((shape_t){*b2->shape, *l1->shape});
  struct tensor *l2 =
      tensor_unop(RELU, tensor_binop(ADD, b2, tensor_matmul(w2, l1)));

  struct tensor *b3 = COL_TENSOR(tensor_nans((shape_t){10}));
  struct tensor *w3 = tensor_nans((shape_t){*b3->shape, *l2->shape});
  struct tensor *l3 =
      tensor_softmax(tensor_binop(ADD, b3, tensor_matmul(w3, l2)));

  struct tensor *x = l0, *yh = l3;
  struct tensor *y = tensor_nans(yh->shape);
  struct node *c = tensor_crossentropy(y, yh);

  struct tensor *w =
      tensor_combine((struct tensor *[]){w1, w2, w3, b1, b2, b3, NULL});

  FILE *c_fp = fopen("mlp.c", "w");
  if (c_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);
  FILE *h_fp = fopen("mlp.h", "w");
  if (h_fp == NULL)
    perror("fopen"), exit(EXIT_FAILURE);

  int visited = 0;
  fprintf(c_fp, "#include \"mlp.h\"\n");
  fprintf(c_fp, "#include <math.h>\n");

  fprintf(h_fp, "typedef double x_t[%zd];\n", shape_size(x->shape));
  fprintf(h_fp, "typedef double w_t[%zd];\n", shape_size(w->shape));
  fprintf(h_fp, "typedef double yh_t[%zd];\n", shape_size(yh->shape));
  fprintf(h_fp, "void predict(x_t x, w_t w, yh_t yh);\n");
  fprintf(c_fp, "void predict(x_t x, w_t w, yh_t yh) {\n");
  TENSOR_FOR(x) fprintf(c_fp, "double t%d = x[%zd];\n", node->id, idx);
  TENSOR_FOR(w) fprintf(c_fp, "double t%d = w[%zd];\n", node->id, idx);
  putc('\n', c_fp);
  ++visited;
  TENSOR_FOR(yh) node_gen(c_fp, "double t%d = ", "t%d", node, visited);
  putc('\n', c_fp);
  TENSOR_FOR(yh) fprintf(c_fp, "yh[%zd] = t%d;\n", idx, node->id);
  fprintf(c_fp, "}\n\n");

  TENSOR_FOR(w) node->grad = LIT(0.0);
  c->grad = LIT(1.0), node_grad(c, ++visited);

  fprintf(h_fp, "typedef double y_t[%zd];\n", shape_size(y->shape));
  fprintf(h_fp, "typedef double dw_t[%zd];\n", shape_size(w->shape));
  fprintf(h_fp, "typedef double *c_t;\n");
  fprintf(h_fp, "void backprop(x_t x, w_t w, y_t y, dw_t dw, c_t c);\n");
  fprintf(c_fp, "void backprop(x_t x, w_t w, y_t y, dw_t dw, c_t c) {\n");
  TENSOR_FOR(x) fprintf(c_fp, "double t%d = x[%zd];\n", node->id, idx);
  TENSOR_FOR(w) fprintf(c_fp, "double t%d = w[%zd];\n", node->id, idx);
  TENSOR_FOR(y) fprintf(c_fp, "double t%d = y[%zd];\n", node->id, idx);
  putc('\n', c_fp);
  node_gen(c_fp, "double t%d = ", "t%d", c, ++visited);
  TENSOR_FOR(w) node_gen(c_fp, "double t%d = ", "t%d", node->grad, visited);
  putc('\n', c_fp);
  fprintf(c_fp, "*c += t%d;\n", c->id);
  TENSOR_FOR(w) fprintf(c_fp, "dw[%zd] += t%d;\n", idx, node->grad->id);
  fprintf(c_fp, "}\n");

  if (fclose(c_fp) == EOF)
    perror("fclose"), exit(EXIT_FAILURE);

  // XXX doc no free
}
