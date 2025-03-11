#include <stddef.h>

typedef size_t shape_t[16];

struct tensor {
  shape_t shape;       // must be null terminated
  struct node *data[]; // length `shape_size(shape)`
};

size_t shape_size(shape_t shape);
size_t shape_rank(shape_t shape);
int shape_cmp(shape_t lhs, shape_t rhs);
struct tensor *tensor_alloc(shape_t shape);
struct tensor *tensor_nans(shape_t shape);
struct tensor *tensor_lit(shape_t shape, struct node *node);
struct tensor *tensor_clone(struct tensor *tensor);
struct tensor *tensor_unop(struct node *(*unop)(struct node *lhs),
                           struct tensor *lhs);
struct tensor *tensor_binop(struct node *(*binop)(struct node *lhs,
                                                  struct node *rhs),
                            struct tensor *lhs, struct tensor *rhs);
struct node *tensor_fold(struct node *id,
                         struct node *(*binop)(struct node *lhs,
                                               struct node *rhs),
                         struct tensor *tensor);
struct tensor *tensor_reshape(shape_t shape, struct tensor *tensor);
struct tensor *tensor_matmul(struct tensor *lhs, struct tensor *rhs);
struct tensor *tensor_combine(struct tensor *tensors[]);
