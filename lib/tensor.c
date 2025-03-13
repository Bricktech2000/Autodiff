#include "tensor.h"
#include "autodiff.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

size_t shape_size(shape_t shape) {
  size_t size = 1;
  for (; *shape; shape++)
    size *= *shape;
  return size;
}

size_t shape_rank(shape_t shape) {
  size_t rank = 0;
  for (; *shape; shape++)
    rank++;
  return rank;
}

int shape_cmp(shape_t lhs, shape_t rhs) {
  while (*lhs && *lhs == *rhs)
    lhs++, rhs++;
  return *lhs - *rhs;
}

struct tensor *tensor_alloc(shape_t shape) {
  struct tensor *tensor =
      calloc(sizeof(*tensor) + shape_size(shape) * sizeof(*tensor->data), 1);
  memcpy(tensor->shape, shape, sizeof(tensor->shape));
  return tensor;
}

struct tensor *tensor_nans(shape_t shape) {
  struct tensor *tensor = tensor_alloc(shape);
  TENSOR_FOR(tensor) node = LIT(NAN);
  return tensor;
}

struct tensor *tensor_lit(shape_t shape, struct node *lit) {
  struct tensor *tensor = tensor_alloc(shape);
  TENSOR_FOR(tensor) node = lit;
  return tensor;
}

struct tensor *tensor_clone(struct tensor *tensor) {
  struct tensor *clone = tensor_alloc(tensor->shape);
  memcpy(clone->shape, tensor->shape, sizeof(clone->shape));
  memcpy(clone->data, tensor->data, shape_size(clone->shape));
  return clone;
}

struct tensor *tensor_unop(struct node *(*unop)(struct node *lhs),
                           struct tensor *lhs) {
  struct tensor *out = tensor_alloc(lhs->shape);
  TENSOR_FOR(out) node = unop(lhs->data[idx]);
  return out;
}

struct tensor *tensor_binop(struct node *(*binop)(struct node *lhs,
                                                  struct node *rhs),
                            struct tensor *lhs, struct tensor *rhs) {
  if (shape_cmp(lhs->shape, rhs->shape) != 0)
    return NULL;

  struct tensor *out = tensor_alloc(lhs->shape);
  TENSOR_FOR(out) node = binop(lhs->data[idx], rhs->data[idx]);
  return out;
}

struct node *tensor_fold(struct node *id,
                         struct node *(*binop)(struct node *lhs,
                                               struct node *rhs),
                         struct tensor *tensor) {
  struct node *acc = id;
  TENSOR_FOR(tensor) acc = binop(acc, node);
  return acc;
}

struct tensor *tensor_reshape(shape_t shape, struct tensor *tensor) {
  if (shape_size(tensor->shape) != shape_size(shape))
    return NULL;

  // in-place reshape
  memcpy(tensor->shape, shape, sizeof(tensor->shape));
  return tensor;
}

struct tensor *tensor_combine(struct tensor *tensors[]) {
  size_t size = 0, j = 0;
  for (struct tensor **tensor = tensors; *tensor; tensor++)
    size += shape_size((*tensor)->shape);

  struct tensor *out = tensor_alloc((shape_t){size});
  for (struct tensor **tensor = tensors; *tensor; tensor++)
    TENSOR_FOR(*tensor) out->data[j++] = node;
  return out;
}

struct tensor *tensor_matmul(struct tensor *lhs, struct tensor *rhs) {
  if (shape_rank(lhs->shape) != 2 || shape_rank(rhs->shape) != 2)
    return NULL;
  if (lhs->shape[1] != rhs->shape[0])
    return NULL;

  struct tensor *out = tensor_alloc((shape_t){lhs->shape[0], rhs->shape[1]});

  struct node *(*lhs_data)[lhs->shape[1]] = &lhs->data;
  struct node *(*rhs_data)[rhs->shape[1]] = &rhs->data;
  struct node *(*out_data)[out->shape[1]] = &out->data;

  for (size_t i = 0; i < lhs->shape[0]; i++) {
    for (size_t k = 0; k < rhs->shape[1]; k++) {
      out_data[i][k] = LIT(0.0);
      for (size_t j = 0; j < lhs->shape[1]; j++)
        out_data[i][k] =
            ADD(out_data[i][k], MUL(lhs_data[i][j], rhs_data[j][k]));
    }
  }

  return out;
}
