#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EMPTY(...)

// clang-format off
#define NODE_TYPES(LEAF, UNOP, BINOP)                                          \
  LEAF(LIT) BINOP(ADD) BINOP(SUB) UNOP(NEG) BINOP(MUL) BINOP(DIV) UNOP(INV)    \
  UNOP(EXP) UNOP(LN) BINOP(POW) BINOP(LOG) UNOP(ID) UNOP(RELU)
// clang-format on

struct node {
#define MKENUM(NAME) NODE_##NAME,
  enum node_type { NODE_TYPES(MKENUM, MKENUM, MKENUM) } type;
  int id, visited;
  struct node *lhs, *rhs;
  struct node *grad, *next;
  double lit;
};

static int node_id = 0;

struct node *LIT(double lit) {
  struct node *node = malloc(sizeof(struct node));
  *node = (struct node){.type = NODE_LIT, .id = node_id++, .lit = lit};
  return node;
}

#define DEF_UNOP(NAME)                                                         \
  struct node *NAME(struct node *lhs) {                                        \
    struct node *node = malloc(sizeof(struct node));                           \
    *node = (struct node){.type = NODE_##NAME, .id = node_id++, .lhs = lhs};   \
    return node;                                                               \
  }

#define DEF_BINOP(NAME)                                                        \
  struct node *NAME(struct node *lhs, struct node *rhs) {                      \
    struct node *node = malloc(sizeof(struct node));                           \
    *node = (struct node){                                                     \
        .type = NODE_##NAME, .id = node_id++, .lhs = lhs, .rhs = rhs};         \
    return node;                                                               \
  }

NODE_TYPES(EMPTY, DEF_UNOP, DEF_BINOP)

// XXX over engineered
void node_dump(struct node *node) {
#define CASE_NODE(NAME, BODY)                                                  \
  case NODE_##NAME:                                                            \
    printf(#NAME "("), BODY, printf(")");                                      \
    break;

#define CASE_LEAF(NAME) CASE_NODE(LIT, printf("%f", node->lit))
#define CASE_UNOP(NAME) CASE_NODE(NAME, node_dump(node->lhs))
#define CASE_BINOP(NAME)                                                       \
  CASE_NODE(NAME, (node_dump(node->lhs), printf(", "), node_dump(node->rhs)))

  switch (node->type) { NODE_TYPES(CASE_LEAF, CASE_UNOP, CASE_BINOP) }
}

// XXX won't work
void node_free(struct node *node) {
  for (struct node *next; node; node = next)
    next = node->next, free(node);
}

void node_revtoposort(struct node *node, struct node **head, int visited) {
  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    node_revtoposort(node->lhs, head, visited);
  if (node->rhs)
    node_revtoposort(node->rhs, head, visited);
  node->next = *head, *head = node;
}

void node_gen(FILE *fp, struct node *node, int visited) {
  if (node->visited == visited)
    return;

  node->visited = visited;
  if (node->lhs)
    node_gen(fp, node->lhs, visited);
  if (node->rhs)
    node_gen(fp, node->rhs, visited);

  if (node->type == NODE_LIT && isnan(node->lit))
    return;

#define NODE_REF(NODE) fprintf(fp, "t%d", NODE->id)
  fprintf(fp, "double t%d = ", node->id);

  switch (node->type) {
  case NODE_LIT:
    fprintf(fp, "%f", node->lit);
    break;
  case NODE_ADD:
    NODE_REF(node->lhs), fprintf(fp, " + "), NODE_REF(node->rhs);
    break;
  case NODE_SUB:
    NODE_REF(node->lhs), fprintf(fp, " - "), NODE_REF(node->rhs);
    break;
  case NODE_NEG:
    fprintf(fp, "-"), NODE_REF(node->lhs);
    break;
  case NODE_MUL:
    NODE_REF(node->lhs), fprintf(fp, " * "), NODE_REF(node->rhs);
    break;
  case NODE_DIV:
    NODE_REF(node->lhs), fprintf(fp, " / "), NODE_REF(node->rhs);
    break;
  case NODE_INV:
    fprintf(fp, "1.0 / "), NODE_REF(node->lhs);
    break;
  case NODE_EXP:
    fprintf(fp, "exp("), NODE_REF(node->lhs), fprintf(fp, ")");
    break;
  case NODE_LN:
    fprintf(fp, "log("), NODE_REF(node->lhs), fprintf(fp, ")");
    break;
  case NODE_POW:
    fprintf(fp, "pow("), NODE_REF(node->lhs), fprintf(fp, ", "),
        NODE_REF(node->rhs), fprintf(fp, ")");
    break;
  case NODE_LOG:
    fprintf(fp, "log("), NODE_REF(node->lhs), fprintf(fp, ") / log("),
        NODE_REF(node->rhs), fprintf(fp, ")");
    break;
  case NODE_ID:
    NODE_REF(node->lhs);
    break;
  case NODE_RELU:
    NODE_REF(node->lhs), fprintf(fp, " < 0.0 ? 0.0 : "), NODE_REF(node->lhs);
    break;
  }

  fprintf(fp, ";\n");
#undef NODE_REF
}

void node_grad(struct node *node, int visited) {
  struct node *head = NULL;
  node_revtoposort(node, &head, visited);

  for (struct node *node = head; node; node = node->next) {
    struct node *lhs_grad = NULL, *rhs_grad = NULL;

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
    case NODE_RELU:
      lhs_grad = DIV(RELU(node->lhs), node->lhs);
      break;
    }

    // chain rule and accumulation
    lhs_grad = MUL(lhs_grad, node->grad);
    if (node->lhs && node->lhs->grad)
      node->lhs->grad = ADD(node->lhs->grad, lhs_grad);
    else if (node->lhs)
      node->lhs->grad = lhs_grad;
    rhs_grad = MUL(rhs_grad, node->grad);
    if (node->rhs && node->rhs->grad)
      node->rhs->grad = ADD(node->rhs->grad, rhs_grad);
    else if (node->rhs)
      node->rhs->grad = rhs_grad;
  }
}
