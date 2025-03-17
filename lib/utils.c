#include "autodiff.h"
#include "tensor.h"
#include <string.h>

struct node *node_square(struct node *node) { return MUL(node, node); }

struct node *node_sigmoid(struct node *x) {
  return INV(ADD(LIT(1.0), EXP(NEG(x))));
}

struct node *node_tanh(struct node *x) {
  struct node *exp_2x = EXP(ADD(x, x));
  return DIV(SUB(exp_2x, LIT(1.0)), ADD(exp_2x, LIT(2.0)));
}

struct node *tensor_sum(bool move_tensor, struct tensor *tensor) {
  return tensor_fold(LIT(0.0), ADD, move_tensor, tensor);
}

struct node *tensor_mean(bool move_tensor, struct tensor *tensor) {
  size_t size = shape_size(tensor->shape);
  return DIV(tensor_sum(move_tensor, tensor), LIT(size));
}

struct node *tensor_mse(bool move_y, struct tensor *y, bool move_yh,
                        struct tensor *yh) {
  return tensor_mean(MOVE tensor_unop(
      node_square, MOVE tensor_binop(SUB, move_y, y, move_yh, yh)));
}

struct node *tensor_r2(bool move_y, struct tensor *y, bool move_yh,
                       struct tensor *yh) {
  struct tensor *yb = tensor_lit(y->shape, tensor_mean(REF y));
  struct node *rss = tensor_sum(MOVE tensor_unop(
      node_square, MOVE tensor_binop(SUB, REF y, move_yh, yh)));
  struct node *tss = tensor_sum(MOVE tensor_unop(
      node_square, MOVE tensor_binop(SUB, move_y, y, MOVE yb)));
  return DIV(rss, tss);
}

struct node *tensor_crossentropy(bool move_y, struct tensor *y, bool move_yh,
                                 struct tensor *yh) {
  return NEG(tensor_sum(
      MOVE tensor_binop(MUL, move_y, y, MOVE tensor_unop(LN, move_yh, yh))));
}

struct tensor *tensor_softmax(bool move_tensor, struct tensor *tensor) {
  shape_t shape;
  memcpy(shape, tensor->shape, sizeof(shape));
  struct tensor *exp = tensor_unop(EXP, move_tensor, tensor);
  struct tensor *sum = tensor_lit(shape, tensor_sum(REF exp));
  return tensor_binop(DIV, MOVE exp, MOVE sum);
}
