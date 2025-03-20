#include <stdbool.h>

struct node *node_id(struct node *node);
struct node *node_double(struct node *node);
struct node *node_square(struct node *node);
struct node *node_relu(struct node *node);
struct node *node_abs(struct node *node);
struct node *node_sigmoid(struct node *node);
struct node *node_tanh(struct node *node);
struct node *tensor_sum(bool move_tensor, struct tensor *tensor);
struct node *tensor_mean(bool move_tensor, struct tensor *tensor);
struct tensor *tensor_sqerr(bool move_y, struct tensor *y, bool move_yh,
                            struct tensor *yh);
struct node *tensor_mse(bool move_y, struct tensor *y, bool move_yh,
                        struct tensor *yh);
struct node *tensor_r2(bool move_y, struct tensor *y, bool move_yh,
                       struct tensor *yh);
struct node *tensor_dot(bool move_lhs, struct tensor *lhs, bool move_rhs,
                        struct tensor *rhs);
struct node *tensor_crossentropy(bool move_y, struct tensor *y, bool move_yh,
                                 struct tensor *yh);
struct tensor *tensor_softmax(bool move_tensor, struct tensor *tensor);
