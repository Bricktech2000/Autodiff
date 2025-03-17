#include "autodiff.h"
#include <math.h>
#include <stdlib.h>

static int node_id = 0;

#define DEF_EMPTY(NAME)
struct node *LIT(double val) {
  struct node *node = malloc(sizeof(*node));
  *node = (struct node){.type = NODE_LIT, .id = node_id++, .val = val};
  return node;
}

#define DEF_UNOP(NAME)                                                         \
  struct node *NAME(struct node *lhs) {                                        \
    struct node *node = malloc(sizeof(*node));                                 \
    *node = (struct node){.type = NODE_##NAME, .id = node_id++, .lhs = lhs};   \
    return node;                                                               \
  }

#define DEF_BINOP(NAME)                                                        \
  struct node *NAME(struct node *lhs, struct node *rhs) {                      \
    struct node *node = malloc(sizeof(*node));                                 \
    *node = (struct node){                                                     \
        .type = NODE_##NAME, .id = node_id++, .lhs = lhs, .rhs = rhs};         \
    return node;                                                               \
  }

NODE_TYPES(DEF_EMPTY, DEF_UNOP, DEF_BINOP)

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
  case NODE_LIT:
    break;
  case NODE_ADD:
    node->val = node->lhs->val + node->rhs->val;
    break;
  case NODE_SUB:
    node->val = node->lhs->val - node->rhs->val;
    break;
  case NODE_NEG:
    node->val = -node->lhs->val;
    break;
  case NODE_MUL:
    node->val = node->lhs->val * node->rhs->val;
    break;
  case NODE_DIV:
    node->val = node->lhs->val / node->rhs->val;
    break;
  case NODE_INV:
    node->val = 1.0 / node->lhs->val;
    break;
  case NODE_EXP:
    node->val = exp(node->lhs->val);
    break;
  case NODE_LN:
    node->val = log(node->lhs->val);
    break;
  case NODE_POW:
    node->val = pow(node->lhs->val, node->rhs->val);
    break;
  case NODE_LOG:
    node->val = log(node->lhs->val) / log(node->rhs->val);
    break;
  case NODE_ID:
    node->val = node->lhs->val;
    break;
  case NODE_ABS:
    node->val = fabs(node->lhs->val);
    break;
  case NODE_RELU:
    node->val = node->lhs->val < 0.0 ? 0.0 : node->lhs->val;
    break;
  }
}

void node_gen(FILE *fp, char *decl, char *ref, struct node *node, int visited) {
  // codegen node into C source code. make sure to call with a unique `visited`

  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    node_gen(fp, decl, ref, node->lhs, visited);
  if (node->rhs)
    node_gen(fp, decl, ref, node->rhs, visited);

  if (node->type == NODE_LIT && isnan(node->val))
    return;

#define REF(NODE) fprintf(fp, ref, NODE->id, NODE->id, NODE->id)
  fprintf(fp, decl, node->id, node->id, node->id);

  switch (node->type) {
  case NODE_LIT:
    fprintf(fp, "%f", node->val);
    break;
  case NODE_ADD:
    REF(node->lhs), fprintf(fp, " + "), REF(node->rhs);
    break;
  case NODE_SUB:
    REF(node->lhs), fprintf(fp, " - "), REF(node->rhs);
    break;
  case NODE_NEG:
    fprintf(fp, "-"), REF(node->lhs);
    break;
  case NODE_MUL:
    REF(node->lhs), fprintf(fp, " * "), REF(node->rhs);
    break;
  case NODE_DIV:
    REF(node->lhs), fprintf(fp, " / "), REF(node->rhs);
    break;
  case NODE_INV:
    fprintf(fp, "1.0 / "), REF(node->lhs);
    break;
  case NODE_EXP:
    fprintf(fp, "exp("), REF(node->lhs), fprintf(fp, ")");
    break;
  case NODE_LN:
    fprintf(fp, "log("), REF(node->lhs), fprintf(fp, ")");
    break;
  case NODE_POW:
    fprintf(fp, "pow("), REF(node->lhs), fprintf(fp, ", "), REF(node->rhs),
        fprintf(fp, ")");
    break;
  case NODE_LOG:
    fprintf(fp, "log("), REF(node->lhs), fprintf(fp, ") / log("),
        REF(node->rhs), fprintf(fp, ")");
    break;
  case NODE_ID:
    REF(node->lhs);
    break;
  case NODE_ABS:
    fprintf(fp, "fabs("), REF(node->lhs), fprintf(fp, ")");
    break;
  case NODE_RELU:
    REF(node->lhs), fprintf(fp, " < 0.0 ? 0.0 : "), REF(node->lhs);
    break;
  }

  fprintf(fp, ";\n");
#undef REF
}

void node_grad(struct node *node, int visited) {
  // compute derivative of `node` and its dependencies with respect to `node`
  // and store results in `grad` fields. before calling make sure that all
  // dependencies' `grad`s hold either `NULL` or `LIT(0.0)` and that `node->
  // grad` is `LIT(1.0)`. make sure to call with a unique `visited`

  struct node *head = NULL;
  revtoposort(node, &head, visited);

  for (struct node *node = head; node; node = node->next) {
    struct node *lhs_grad = NULL, *rhs_grad = NULL;

    // derivative of `node->lhs` and `node->rhs` with respect to `node`
    switch (node->type) {
    case NODE_LIT:
      break;
    case NODE_ADD:
      lhs_grad = LIT(1.0);
      rhs_grad = lhs_grad;
      break;
    case NODE_SUB:
      lhs_grad = LIT(1.0);
      rhs_grad = LIT(-1.0);
      break;
    case NODE_NEG:
      lhs_grad = LIT(-1.0);
      break;
    case NODE_MUL:
      lhs_grad = node->rhs;
      rhs_grad = node->lhs;
      break;
    case NODE_DIV:
      lhs_grad = INV(node->rhs);
      rhs_grad = NEG(DIV(node, node->rhs));
      break;
    case NODE_INV:
      lhs_grad = NEG(DIV(node, node->lhs));
      break;
    case NODE_EXP:
      lhs_grad = node;
      break;
    case NODE_LN:
      lhs_grad = INV(node->lhs);
      break;
    case NODE_POW:
      lhs_grad = MUL(node->rhs, DIV(node, node->lhs));
      rhs_grad = MUL(node, LN(node->lhs));
      break;
    case NODE_LOG:;
      struct node *ln_rhs = LN(node->rhs);
      lhs_grad = INV(MUL(node->lhs, ln_rhs));
      rhs_grad = NEG(DIV(node, MUL(node->rhs, ln_rhs)));
      break;
    case NODE_ID:
      lhs_grad = LIT(1.0);
      break;
    case NODE_ABS:
      lhs_grad = DIV(ABS(node->lhs), node->lhs);
      break;
    case NODE_RELU:
      lhs_grad = DIV(RELU(node->lhs), node->lhs);
      break;
    }

    // chain rule and accumulation
    if (node->lhs && node->lhs->grad)
      node->lhs->grad = ADD(node->lhs->grad, MUL(lhs_grad, node->grad));
    else if (node->lhs)
      node->lhs->grad = MUL(lhs_grad, node->grad);
    if (node->rhs && node->rhs->grad)
      node->rhs->grad = ADD(node->rhs->grad, MUL(rhs_grad, node->grad));
    else if (node->rhs)
      node->rhs->grad = MUL(rhs_grad, node->grad);
  }
}
