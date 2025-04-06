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

int node_mark(struct node *node, struct node **head, int count, int visited) {
  // mark `node` and its dependencies as `visited` and store them in the
  // linked list formed by `next` fields starting at `head` in reverse
  // topological order. pass in `head = NULL` to discard the linked list.
  // call with `count = 0`. returns the number of nodes marked

  if (node->visited == visited)
    return count;

  node->visited = visited, count++;
  if (node->lhs)
    count = node_mark(node->lhs, head, count, visited);
  if (node->rhs)
    count = node_mark(node->rhs, head, count, visited);

  if (head != NULL)
    node->next = *head, *head = node;

  return count;
}

void node_free(struct node *head, int visited) {
  // free the nodes in the linked list formed by `next` fields starting at
  // `head`, including their gradients and gradients' dependencies
  node_zerograd(head, visited);
  for (struct node *next; head; head = next)
    next = head->next, free(head);
}

void node_zerograd(struct node *head, int visited) {
  // free the gradients, and their dependencies, of the nodes in the linked
  // list formed by `next` fields starting at `head`
  struct node *grads = NULL;
  for (; head; head = head->next)
    if (head->grad)
      node_mark(head->grad, &grads, 0, visited), head->grad = NULL;
  if (grads)
    node_free(grads, visited);
}

void node_codegen(FILE *fp, char *decl_fmt, char *ref_fmt, struct node *node,
                  int visited) {
  // codegen node into C source code. `decl_fmt` and `ref_fmt` are format
  // strings that specify how to declare temporaries and refer to temporaries,
  // respectively. codegens nothing for `node_lit(NAN)`s, so they can be used
  // as inputs. make sure to call with a unique `visited`

  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    node_codegen(fp, decl_fmt, ref_fmt, node->lhs, visited);
  if (node->rhs)
    node_codegen(fp, decl_fmt, ref_fmt, node->rhs, visited);

  if (node->type == NODE_LIT && isnan(node->val))
    return;

    // passing in `node->id` several times so that the `ref_fmt` and `decl_fmt`
    // format strings can refer to a node's ID several times if needed
#define GEN_REF(NODE) fprintf(fp, ref_fmt, NODE->id, NODE->id, NODE->id)
  fprintf(fp, decl_fmt, node->id, node->id, node->id);

  switch (node->type) {
    // see runtime.h
#define GEN_LIT(UC, LC)                                                        \
  case NODE_##UC:                                                              \
    /* using the conversion specifier 'A' in hopes of producing `INFINITY` and \
     * `NAN` for <math.h> would be futile because it is implementation-defined \
     * whether infinity converts to 'INF' or to 'INFINITY' and whether NaN     \
     * converts to 'NAN' or to 'NAN(n-char-sequence)'; see ISO/IEC 9899:TC3,   \
     * $7.19.6.1, paragraph 8, conversion specifiers 'f,F' */                  \
    fprintf(fp, "op_" #LC "(%#a)", node->val);                                 \
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

void node_grad(struct node *node, int visited) {
  // compute derivative of `node` and its dependencies with respect to `node`
  // and store results in `grad` fields. before calling make sure that all
  // dependencies' `grad`s hold either `NULL` or `node_lit(0.0)` and that
  // `node->grad` is `node_lit(1.0)`. make sure to call with a unique `visited`

  struct node *head = NULL;
  node_mark(node, &head, 0, visited); // reverse topological order

  for (; head; head = head->next) {
    struct node *lhs_grad = NULL, *rhs_grad = NULL;

    // derivatives of `head->lhs` and `head->rhs` with respect to `head`
    switch (head->type) {
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
      lhs_grad = head->rhs;
      rhs_grad = head->lhs;
      break;
    case NODE_DIV:
      lhs_grad = node_inv(head->rhs);
      rhs_grad = node_neg(node_mul(head, lhs_grad));
      break;
    case NODE_INV:
      lhs_grad = node_neg(node_div(head, head->lhs));
      break;
    case NODE_EXP:
      lhs_grad = head;
      break;
    case NODE_LOG:
      lhs_grad = node_inv(head->lhs);
      break;
    case NODE_EXP2:
      lhs_grad = node_mul(head, node_log(head->lhs));
      break;
    case NODE_LOG2:
      lhs_grad = node_inv(node_mul(head->lhs, node_lit(log(2.0))));
      break;
    case NODE_POW:
      lhs_grad = node_mul(head->rhs, node_div(head, head->lhs));
      rhs_grad = node_mul(head, node_log(head->lhs));
      break;
    case NODE_SQRT:
      lhs_grad = node_inv(node_mul(node_lit(2.0), head));
      break;
    case NODE_CBRT:
      lhs_grad = node_div(head, node_mul(node_lit(3.0), head->lhs));
      break;
    case NODE_MIN:;
      struct node *sub_rhs_lhs = node_sub(head->rhs, head->lhs);
      lhs_grad = node_div(node_relu(sub_rhs_lhs), sub_rhs_lhs);
      rhs_grad = node_sub(node_lit(1.0), lhs_grad);
      break;
    case NODE_MAX:;
      struct node *sub_lhs_rhs = node_sub(head->lhs, head->rhs);
      lhs_grad = node_div(node_relu(sub_lhs_rhs), sub_lhs_rhs);
      rhs_grad = node_sub(node_lit(1.0), lhs_grad);
      break;
    case NODE_ABS:
      lhs_grad = node_div(head, head->lhs);
      break;
    case NODE_RELU:
      lhs_grad = node_div(head, head->lhs);
      break;
    }

    // int chain = head->grad->type != NODE_LIT || head->grad->val != 1.0;

    if (head->lhs) {
      lhs_grad = node_mul(lhs_grad, head->grad); // chain rule
      head->lhs->grad = head->lhs->grad ? node_add(head->lhs->grad, lhs_grad)
                                        : lhs_grad; // gradient accumulation
    }
    if (head->rhs) {
      rhs_grad = node_mul(rhs_grad, head->grad); // chain rule
      head->rhs->grad = head->rhs->grad ? node_add(head->rhs->grad, rhs_grad)
                                        : rhs_grad; // gradient accumulation
    }
  }
}
