struct node *node_square(struct node *node);
struct node *node_sigmoid(struct node *x);
struct node *node_tanh(struct node *x);
struct node *tensor_sum(struct tensor *tensor);
struct node *tensor_mean(struct tensor *tensor);
struct node *tensor_mse(struct tensor *y, struct tensor *yh);
struct node *tensor_r2(struct tensor *y, struct tensor *yh);
struct node *tensor_crossentropy(struct tensor *y, struct tensor *yh);
struct tensor *tensor_softmax(struct tensor *tensor);
