#include "autodiff.h"
#include "tensor.h"

struct node *node_square(struct node *node) { return MUL(node, node); }

struct node *node_sigmoid(struct node *x) {
  return INV(ADD(LIT(1.0), EXP(NEG(x))));
}

struct node *node_tanh(struct node *x) {
  struct node *exp_2x = EXP(ADD(x, x));
  return DIV(SUB(exp_2x, LIT(1.0)), ADD(exp_2x, LIT(2.0)));
}

struct node *tensor_sum(struct tensor *tensor) {
  return tensor_fold(LIT(0.0), ADD, tensor);
}

struct node *tensor_mean(struct tensor *tensor) {
  return DIV(tensor_sum(tensor), LIT(shape_size(tensor->shape)));
}

struct node *tensor_mse(struct tensor *y, struct tensor *yh) {
  return tensor_mean(tensor_unop(node_square, tensor_binop(SUB, y, yh)));
}

struct node *tensor_r2(struct tensor *y, struct tensor *yh) {
  struct tensor *yb = tensor_lit(y->shape, tensor_mean(y));
  return DIV(tensor_sum(tensor_unop(node_square, tensor_binop(SUB, y, yh))),
             tensor_sum(tensor_unop(node_square, tensor_binop(SUB, y, yb))));
}

struct node *tensor_crossentropy(struct tensor *y, struct tensor *yh) {
  return NEG(tensor_sum(tensor_binop(MUL, y, tensor_unop(LN, yh))));
}

struct tensor *tensor_softmax(struct tensor *tensor) {
  struct tensor *exp = tensor_unop(EXP, tensor);
  return tensor_binop(DIV, exp, tensor_lit(tensor->shape, tensor_sum(exp)));
}
