#include "autodiff.h"
#include "runtime.h"
#include <stdlib.h>

static int node_id = 0;

#define DEF_LIT(UC, LC)                                                        \
  struct node *node_##LC(double val) {                                         \
    struct node *node = malloc(sizeof(*node));                                 \
    *node = (struct node){.type = NODE_##UC, .id = node_id++, .val = val};     \
    return node;                                                               \
  }

#define DEF_UNOP(UC, LC)                                                       \
  struct node *node_##LC(struct node *lhs) {                                   \
    struct node *node = malloc(sizeof(*node));                                 \
    *node = (struct node){.type = NODE_##UC, .id = node_id++, .lhs = lhs};     \
    return node;                                                               \
  }

#define DEF_BINOP(UC, LC)                                                      \
  struct node *node_##LC(struct node *lhs, struct node *rhs) {                 \
    struct node *node = malloc(sizeof(*node));                                 \
    *node = (struct node){                                                     \
        .type = NODE_##UC, .id = node_id++, .lhs = lhs, .rhs = rhs};           \
    return node;                                                               \
  }

NODE_TYPES(DEF_LIT, DEF_UNOP, DEF_BINOP)

#undef DEF_LIT
#undef DEF_UNOP
#undef DEF_BINOP

static void revtoposort(struct node *node, struct node **head, int visited) {
  // compute reverse topolgical sorting of `node` and store result in the linked
  // list formed by `next` fields starting at `head`

  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    revtoposort(node->lhs, head, visited);
  if (node->rhs)
    revtoposort(node->rhs, head, visited);
  node->next = *head, *head = node;
}

void node_free(struct node *head, int visited) {
  // free linked list of nodes formed by `next` fields starting at `head`, and
  // their dependencies, and their gradients. clobbers `next` fields. make sure
  // to call with a unique `visited`

  struct node *tail = &(struct node){.next = head};
  while (tail->next)
    tail = tail->next, tail->visited = visited;

  for (struct node *node = head; node; node = node->next) {
    if (node->lhs && node->lhs->visited != visited)
      tail->next = node->lhs, tail = tail->next, tail->visited = visited;
    if (node->rhs && node->rhs->visited != visited)
      tail->next = node->rhs, tail = tail->next, tail->visited = visited;
    if (node->grad && node->grad->visited != visited)
      tail->next = node->grad, tail = tail->next, tail->visited = visited;
    tail->next = NULL;
  }

  for (struct node *next, *node = head; node; node = next)
    next = node->next, free(node);
}

void node_eval(struct node *node, int visited) {
  // evaluate the value of `node` and its dependencies and store results in
  // `val` fields. make sure all dependencies of type `NODE_LIT` actually
  // hold a literal in their `val`. make sure to call with a unique `visited`

  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    node_eval(node->lhs, visited);
  if (node->rhs)
    node_eval(node->rhs, visited);

  switch (node->type) {
    // see runtime.h
#define EVAL_LIT(UC, LC)                                                       \
  case NODE_##UC:                                                              \
    node->val = op_##LC(node->val);                                            \
    break;
#define EVAL_UNOP(UC, LC)                                                      \
  case NODE_##UC:                                                              \
    node->val = op_##LC(node->lhs->val);                                       \
    break;
#define EVAL_BINOP(UC, LC)                                                     \
  case NODE_##UC:                                                              \
    node->val = op_##LC(node->lhs->val, node->rhs->val);                       \
    break;

    NODE_TYPES(EVAL_LIT, EVAL_UNOP, EVAL_BINOP)

#undef EVAL_LIT
#undef EVAL_UNOP
#undef EVAL_BINOP
  }
}

void node_gen(FILE *fp, char *decl, char *ref, struct node *node, int visited) {
  // codegen node into C source code. `decl` and `ref` are format strings that
  // describe how to declare temporaries and refer to temporaries, respectively.
  // codegens nothing for `node_lit(NAN)`s, so they can be used as inputs. make
  // sure to call with a unique `visited`

  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    node_gen(fp, decl, ref, node->lhs, visited);
  if (node->rhs)
    node_gen(fp, decl, ref, node->rhs, visited);

  if (node->type == NODE_LIT && isnan(node->val))
    return;

    // passing in `node->id` several times so that the `ref` and `decl` format
    // strings can refer to a node's ID several times if needed
#define GEN_REF(NODE) fprintf(fp, ref, NODE->id, NODE->id, NODE->id)
  fprintf(fp, decl, node->id, node->id, node->id);

  switch (node->type) {
    // see runtime.h
#define GEN_LIT(UC, LC)                                                        \
  case NODE_##UC:                                                              \
    fprintf(fp, "op_" #LC "(%f)", node->val);                                  \
    break;
#define GEN_UNOP(UC, LC)                                                       \
  case NODE_##UC:                                                              \
    fprintf(fp, "op_" #LC "("), GEN_REF(node->lhs), fprintf(fp, ")");          \
    break;
#define GEN_BINOP(UC, LC)                                                      \
  case NODE_##UC:                                                              \
    fprintf(fp, "op_" #LC "("), GEN_REF(node->lhs), fprintf(fp, ", "),         \
        GEN_REF(node->rhs), fprintf(fp, ")");                                  \
    break;

    NODE_TYPES(GEN_LIT, GEN_UNOP, GEN_BINOP)

#undef GEN_LIT
#undef GEN_UNOP
#undef GEN_BINOP
  }

  fprintf(fp, ";\n");
#undef GEN_REF
}

void node_grad(struct node *node, int visited) {
  // compute derivative of `node` and its dependencies with respect to `node`
  // and store results in `grad` fields. before calling make sure that all
  // dependencies' `grad`s hold either `NULL` or `node_lit(0.0)` and that
  // `node->grad` is `node_lit(1.0)`. make sure to call with a unique `visited`

  struct node *head = NULL;
  revtoposort(node, &head, visited);

  for (struct node *node = head; node; node = node->next) {
    struct node *lhs_grad = NULL, *rhs_grad = NULL;

    // derivatives of `node->lhs` and `node->rhs` with respect to `node`
    switch (node->type) {
    case NODE_LIT:
      break;
    case NODE_ADD:
      lhs_grad = node_lit(1.0);
      rhs_grad = lhs_grad;
      break;
    case NODE_SUB:
      lhs_grad = node_lit(1.0);
      rhs_grad = node_lit(-1.0);
      break;
    case NODE_NEG:
      lhs_grad = node_lit(-1.0);
      break;
    case NODE_MUL:
      lhs_grad = node->rhs;
      rhs_grad = node->lhs;
      break;
    case NODE_DIV:
      lhs_grad = node_inv(node->rhs);
      rhs_grad = node_neg(node_div(node, node->rhs));
      break;
    case NODE_INV:
      lhs_grad = node_neg(node_div(node, node->lhs));
      break;
    case NODE_EXP:
      lhs_grad = node;
      break;
    case NODE_LOG:
      lhs_grad = node_inv(node->lhs);
      break;
    case NODE_EXP2:
      lhs_grad = node_mul(node, node_log(node->lhs));
      break;
    case NODE_LOG2:
      lhs_grad = node_inv(node_mul(node->lhs, node_lit(log(2.0))));
      break;
    case NODE_POW:
      lhs_grad = node_mul(node->rhs, node_div(node, node->lhs));
      rhs_grad = node_mul(node, node_log(node->lhs));
      break;
    case NODE_SQRT:
      lhs_grad = node_inv(node_mul(node_lit(2.0), node));
      break;
    case NODE_CBRT:
      lhs_grad = node_div(node, node_mul(node_lit(3.0), node->lhs));
      break;
    case NODE_MIN:;
      struct node *sub_rhs_lhs = node_sub(node->rhs, node->lhs);
      lhs_grad = node_div(node_relu(sub_rhs_lhs), sub_rhs_lhs);
      rhs_grad = node_sub(node_lit(1.0), lhs_grad);
      break;
    case NODE_MAX:;
      struct node *sub_lhs_rhs = node_sub(node->lhs, node->rhs);
      lhs_grad = node_div(node_relu(sub_lhs_rhs), sub_lhs_rhs);
      rhs_grad = node_sub(node_lit(1.0), lhs_grad);
      break;
    case NODE_ABS:
      lhs_grad = node_div(node, node->lhs);
      break;
    case NODE_RELU:
      lhs_grad = node_div(node, node->lhs);
      break;
    }

    if (node->lhs) {
      lhs_grad = node_mul(lhs_grad, node->grad); // chain rule
      node->lhs->grad = node->lhs->grad ? node_add(node->lhs->grad, lhs_grad)
                                        : lhs_grad; // gradient accumulation
    }
    if (node->rhs) {
      rhs_grad = node_mul(rhs_grad, node->grad); // chain rule
      node->rhs->grad = node->rhs->grad ? node_add(node->rhs->grad, rhs_grad)
                                        : rhs_grad; // gradient accumulation
    }
  }
}
