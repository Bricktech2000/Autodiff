#include <stdio.h>

// clang-format off
#define NODE_TYPES(LEAF, UNOP, BINOP)                                          \
  LEAF(LIT) BINOP(ADD) BINOP(SUB) UNOP(NEG) BINOP(MUL) BINOP(DIV) UNOP(INV)    \
  UNOP(EXP) UNOP(LN) BINOP(POW) BINOP(LOG) UNOP(ID) UNOP(ABS) UNOP(RELU)
// clang-format on

struct node {
#define MKENUM(NAME) NODE_##NAME,
  enum node_type { NODE_TYPES(MKENUM, MKENUM, MKENUM) } type;
  int id, visited;        // for node graph traversal
  struct node *lhs, *rhs; // child nodes; may be `NULL` depending on `type`
  struct node *next;      // for output of `rev_toposort`
  struct node *grad;      // for output of `node_grad`
  double val;             // for output of `node_eval`
};

#define DECL_EMPTY(NAME)
struct node *LIT(double val);
#define DECL_UNOP(NAME) struct node *NAME(struct node *lhs);
#define DECL_BINOP(NAME) struct node *NAME(struct node *lhs, struct node *rhs);

NODE_TYPES(DECL_EMPTY, DECL_UNOP, DECL_BINOP)

void node_eval(struct node *node, int visited);
void node_gen(FILE *fp, char *decl, char *ref, struct node *node, int visited);
void node_grad(struct node *node, int visited);
