#include "lib/autodiff.h"
#include "lib/tensor.h"

#define ROW_TENSOR(TENSOR)                                                     \
  tensor_reshape((shape_t){1, TENSOR->shape[0]}, TENSOR)
#define COL_TENSOR(TENSOR)                                                     \
  tensor_reshape((shape_t){TENSOR->shape[0], 1}, TENSOR)

#define TENSOR_FOR(TENSOR)                                                     \
  for (size_t idx = 0; idx < shape_size(TENSOR->shape); idx++)                 \
    for (struct node *node = TENSOR->data[idx]; node;                          \
         TENSOR->data[idx] = node, node = NULL)

static struct node *node_square(struct node *node) { return MUL(node, node); }

static struct node *node_sigmoid(struct node *x) {
  return INV(ADD(LIT(1.0), EXP(NEG(x))));
}

static struct node *node_tanh(struct node *x) {
  struct node *exp_2x = EXP(ADD(x, x));
  return DIV(SUB(exp_2x, LIT(1.0)), ADD(exp_2x, LIT(2.0)));
}

static struct node *tensor_sum(struct tensor *tensor) {
  return tensor_fold(LIT(0.0), ADD, tensor);
}

static struct node *tensor_mean(struct tensor *tensor) {
  return DIV(tensor_sum(tensor), LIT(shape_size(tensor->shape)));
}

static struct node *tensor_mse(struct tensor *y, struct tensor *yh) {
  return tensor_mean(tensor_unop(node_square, tensor_binop(SUB, y, yh)));
}

static struct node *tensor_r2(struct tensor *y, struct tensor *yh) {
  struct tensor *yb = tensor_lit(y->shape, tensor_mean(y));
  return DIV(tensor_sum(tensor_unop(node_square, tensor_binop(SUB, y, yh))),
             tensor_sum(tensor_unop(node_square, tensor_binop(SUB, y, yb))));
}

static struct node *tensor_crossentropy(struct tensor *y, struct tensor *yh) {
  return NEG(tensor_sum(tensor_binop(MUL, y, tensor_unop(LN, yh))));
}

static struct tensor *tensor_softmax(struct tensor *tensor) {
  struct tensor *exp = tensor_unop(EXP, tensor);
  return tensor_binop(DIV, exp, tensor_lit(tensor->shape, tensor_sum(exp)));
}
