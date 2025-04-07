#include "tensor.h"
#include "autodiff.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

size_t shape_size(size_t *shape) {
  size_t size = 1;
  for (; *shape; shape++)
    size *= *shape;
  return size;
}

size_t shape_rank(size_t *shape) {
  size_t rank = 0;
  for (; *shape; shape++)
    rank++;
  return rank;
}

int shape_cmp(size_t *lhs, size_t *rhs) {
  while (*lhs && *lhs == *rhs)
    lhs++, rhs++;
  return *lhs - *rhs;
}

struct tensor tensor_alloc(shape_t shape) {
  struct tensor tensor;
  memcpy(tensor.shape, shape, sizeof(tensor.shape));
  tensor.data = calloc(shape_size(shape), sizeof(*tensor.data));
  return tensor;
}

struct tensor tensor_nans(shape_t shape) {
  struct tensor tensor = tensor_alloc(shape);
  TENSOR_FOR(tensor) node = node_lit(NAN);
  return tensor;
}

struct tensor tensor_repeat(shape_t shape, struct node *item) {
  struct tensor tensor = tensor_alloc(shape);
  TENSOR_FOR(tensor) node = item;
  return tensor;
}

struct tensor tensor_clone(bool move_tensor, struct tensor tensor) {
  if (move_tensor)
    return tensor;

  struct tensor clone = tensor_alloc(tensor.shape);
  memcpy(clone.shape, tensor.shape, sizeof(clone.shape));
  memcpy(clone.data, tensor.data, shape_size(clone.shape));
  return clone;
}

struct tensor tensor_unop(struct node *(*unop)(struct node *lhs), bool move_lhs,
                          struct tensor lhs) {
  struct tensor out = move_lhs ? lhs : tensor_alloc(lhs.shape);
  TENSOR_FOR(out) node = unop(lhs.data[idx]);
  return out;
}

struct tensor tensor_binop(struct node *(*binop)(struct node *lhs,
                                                 struct node *rhs),
                           bool move_lhs, struct tensor lhs, bool move_rhs,
                           struct tensor rhs) {
  if (shape_cmp(lhs.shape, rhs.shape) != 0)
    abort();

  struct tensor out = move_lhs ? lhs : move_rhs ? rhs : tensor_alloc(lhs.shape);
  TENSOR_FOR(out) node = binop(lhs.data[idx], rhs.data[idx]);
  if (move_lhs && move_rhs)
    free(rhs.data);
  return out;
}

struct node *tensor_fold(struct node *id,
                         struct node *(*binop)(struct node *lhs,
                                               struct node *rhs),
                         bool move_tensor, struct tensor tensor) {
  struct node *acc = id;
  TENSOR_FOR(tensor) acc = binop(acc, node);
  if (move_tensor)
    free(tensor.data);
  return acc;
}

struct tensor tensor_matmul(bool move_lhs, struct tensor lhs, bool move_rhs,
                            struct tensor rhs) {
  // perform matrix multiplication on the two outermost dimensions, operating
  // element-wise on inner dimensions

  if (shape_rank(lhs.shape) < 2 || shape_rank(rhs.shape) < 2)
    abort();
  if (shape_cmp(lhs.shape + 2, rhs.shape + 2) != 0)
    abort();
  if (lhs.shape[1] != rhs.shape[0])
    abort();

  shape_t shape = {lhs.shape[0]};
  memcpy(shape + 1, rhs.shape + 1, sizeof(rhs.shape) - sizeof(*rhs.shape));
  struct tensor out = tensor_repeat(shape, node_lit(0.0));

  for (size_t i = 0; i < lhs.shape[0]; i++) {
    for (size_t k = 0; k < rhs.shape[1]; k++) {
      struct tensor out_slice = tensor_slice(REF tensor_slice(REF out, i), k);
      for (size_t j = 0; j < lhs.shape[1]; j++) {
        struct tensor mul_lhs_rhs = tensor_binop(
            node_mul, REF tensor_slice(REF tensor_slice(REF lhs, i), j),
            REF tensor_slice(REF tensor_slice(REF rhs, j), k));
        struct tensor ret =
            tensor_binop(node_add, MOVE out_slice, MOVE mul_lhs_rhs);
        if (ret.data != out_slice.data) // hehe
          abort();
      }
    }
  }

  if (move_lhs)
    free(lhs.data);
  if (move_rhs)
    free(rhs.data);

  return out;
}

struct tensor tensor_reshape(shape_t shape, bool move_tensor,
                             struct tensor tensor) {
  // if `tensor` is lent, returns a borrowed tensor of the same lifetime. if
  // `tensor` is moved in, returns an owned tensor

  if (shape_size(tensor.shape) != shape_size(shape))
    abort();

  (void)move_tensor;
  memcpy(tensor.shape, shape, sizeof(tensor.shape));
  return tensor;
}

struct tensor tensor_slice(bool move_tensor, struct tensor tensor, size_t idx) {
  // `tensor` must be lent. returns a borrowed tensor of the same lifetime

  if (idx >= *tensor.shape)
    abort();
  if (move_tensor)
    abort();

  struct tensor slice = {0};
  memcpy(slice.shape, tensor.shape + 1,
         sizeof(tensor.shape) - sizeof(*tensor.shape));
  slice.data = tensor.data + idx * shape_size(slice.shape);
  return slice;
}

struct tensor tensor_subscript(bool move_tensor, struct tensor tensor,
                               size_t idx) {
  struct tensor subscript = tensor_clone(REF tensor_slice(REF tensor, idx));
  if (move_tensor)
    free(tensor.data);
  return subscript;
}

struct tensor tensor_collect(bool move_tensors[], struct tensor tensors[]) {
  // collect a bunch of tensors of disparate shapes into a single tensor of rank
  // one, to aid in iterating over all nodes

  size_t size = 0, j = 0;
  for (struct tensor *tensor = tensors; tensor->data; tensor++)
    size += shape_size(tensor->shape);

  struct tensor collected = tensor_alloc((shape_t){size});
  for (struct tensor *tensor = tensors; tensor->data; tensor++) {
    TENSOR_FOR(*tensor) collected.data[j++] = node;
    if (move_tensors[tensor - tensors])
      free(tensor->data);
  }

  return collected;
}
