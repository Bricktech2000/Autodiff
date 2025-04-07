static struct node *node_id(struct node *node) { return node; }

static struct node *node_double(struct node *node) {
  return node_add(node, node);
}

static struct node *node_triple(struct node *node) {
  return node_add(node, node_double(node));
}

static struct node *node_square(struct node *node) {
  return node_mul(node, node);
}

static struct node *node_cube(struct node *node) {
  return node_mul(node, node_square(node));
}

static struct node *node_sigmoid(struct node *node) {
  return node_inv(node_add(node_lit(1.0), node_exp(node_neg(node))));
}

static struct node *node_tanh(struct node *node) {
  struct node *exp_2x = node_exp(node_double(node));
  return node_div(node_sub(exp_2x, node_lit(1.0)),
                  node_add(exp_2x, node_lit(2.0)));
}

static struct node *tensor_sum(bool move_tensor, struct tensor tensor) {
  return tensor_fold(node_lit(0.0), node_add, move_tensor, tensor);
}

static struct node *tensor_mean(bool move_tensor, struct tensor tensor) {
  size_t size = shape_size(tensor.shape);
  return node_div(tensor_sum(move_tensor, tensor), node_lit(size));
}

static struct tensor tensor_sqerr(bool move_y, struct tensor y, bool move_yh,
                                  struct tensor yh) {
  return tensor_unop(node_square,
                     MOVE tensor_binop(node_sub, move_y, y, move_yh, yh));
}

static struct node *tensor_mse(bool move_y, struct tensor y, bool move_yh,
                               struct tensor yh) {
  return tensor_mean(MOVE tensor_sqerr(move_y, y, move_yh, yh));
}

static struct node *tensor_rmse(bool move_y, struct tensor y, bool move_yh,
                                struct tensor yh) {
  return node_sqrt(tensor_mse(move_y, y, move_yh, yh));
}

static struct node *tensor_r2(bool move_y, struct tensor y, bool move_yh,
                              struct tensor yh) {
  struct tensor yb = tensor_repeat(y.shape, tensor_mean(REF y));
  return node_div(tensor_sum(MOVE tensor_sqerr(REF y, move_yh, yh)),
                  tensor_sum(MOVE tensor_sqerr(move_y, y, MOVE yb)));
}

static struct node *tensor_dot(bool move_lhs, struct tensor lhs, bool move_rhs,
                               struct tensor rhs) {
  return tensor_sum(MOVE tensor_binop(node_mul, move_lhs, lhs, move_rhs, rhs));
}

static struct node *tensor_crossentropy(bool move_y, struct tensor y,
                                        bool move_yh, struct tensor yh) {
  return node_neg(
      tensor_dot(move_y, y, MOVE tensor_unop(node_log, move_yh, yh)));
}

static struct tensor tensor_softmax(bool move_tensor, struct tensor tensor) {
  struct tensor exp = tensor_unop(node_exp, move_tensor, tensor);
  struct tensor sum = tensor_repeat(exp.shape, tensor_sum(REF exp));
  return tensor_binop(node_div, MOVE exp, MOVE sum);
}

static struct tensor row_tensor(bool move_tensor, struct tensor tensor) {
  // has the ownership semantics of `tensor_reshape`
  return tensor_reshape((shape_t){1, *tensor.shape}, move_tensor, tensor);
}

static struct tensor col_tensor(bool move_tensor, struct tensor tensor) {
  // has the ownership semantics of `tensor_reshape`
  return tensor_reshape((shape_t){*tensor.shape, 1}, move_tensor, tensor);
}
