#include <stdbool.h>
#include <stddef.h>

// intended for `move...` parameters. `MOVE` transfers ownership of its
// tensor to the callee while `REF` lends it. returned tensors are owned,
// unless otherwise specified
#define MOVE true,
#define REF false,

#define TENSOR_FOR(TENSOR)                                                     \
  for (size_t idx = 0; idx < shape_size((TENSOR).shape); idx++)                \
    for (struct node *node = (TENSOR).data[idx], **_p = &node; _p;             \
         (TENSOR).data[idx] = node, _p = NULL)

typedef size_t shape_t[16];

struct tensor {
  shape_t shape;      // must be null terminated
  struct node **data; // array of length `shape_size(shape)`
};

size_t shape_size(size_t *shape);
size_t shape_rank(size_t *shape);
int shape_cmp(size_t *lhs, size_t *rhs);
struct tensor tensor_alloc(shape_t shape);
struct tensor tensor_nans(shape_t shape);
struct tensor tensor_repeat(shape_t shape, struct node *item);
struct tensor tensor_clone(bool move_tensor, struct tensor tensor);
struct tensor tensor_unop(struct node *(*unop)(struct node *lhs), bool move_lhs,
                          struct tensor lhs);
struct tensor tensor_binop(struct node *(*binop)(struct node *lhs,
                                                 struct node *rhs),
                           bool move_lhs, struct tensor lhs, bool move_rhs,
                           struct tensor rhs);
struct node *tensor_fold(struct node *id,
                         struct node *(*binop)(struct node *lhs,
                                               struct node *rhs),
                         bool move_tensor, struct tensor tensor);
struct tensor tensor_matmul(bool move_lhs, struct tensor lhs, bool move_rhs,
                            struct tensor rhs);
struct tensor tensor_reshape(shape_t shape, bool move_tensor,
                             struct tensor tensor);
struct tensor tensor_slice(bool move_tensor, struct tensor tensor, size_t idx);
struct tensor tensor_subscript(bool move_tensor, struct tensor tensor,
                               size_t idx);
struct tensor tensor_collect(bool move_tensors[], struct tensor tensors[]);
