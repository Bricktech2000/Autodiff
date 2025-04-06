#include <stdio.h>

// the main criterion all node types should meet is that their implementation
// (see runtime.h) should be, up to partial application, either a library call
// to <math.h> or a builtin operator, reasoning being that the set of floating-
// point primitives provided by the C language is probably a well-balanced one.
// clang-format off
#define NODE_TYPES(LIT_, UNOP, BINOP)                                          \
  LIT_(LIT, lit)                                                               \
  BINOP(ADD, add) BINOP(SUB, sub) UNOP(NEG, neg)                               \
  BINOP(MUL, mul) BINOP(DIV, div) UNOP(INV, inv)                               \
  UNOP(EXP, exp) UNOP(LOG, log) UNOP(EXP2, exp2) UNOP(LOG2, log2)              \
  BINOP(POW, pow) UNOP(SQRT, sqrt) UNOP(CBRT, cbrt)                            \
  BINOP(MIN, min) BINOP(MAX, max) UNOP(RELU, relu) UNOP(ABS, abs)
// clang-format on

struct node {
#define MKENUM(UC, LC) NODE_##UC,
  enum node_type { NODE_TYPES(MKENUM, MKENUM, MKENUM) } type;
#undef MKENUM
  int id, visited;        // for node graph traversal
  struct node *lhs, *rhs; // child nodes; may be `NULL` depending on `type`
  struct node *next;      // for output of `node_mark`
  struct node *grad;      // for output of `node_grad`
  double val;             // for output of `node_eval`
};

#define DECL_LIT(UL, LC) struct node *node_##LC(double val);
#define DECL_UNOP(UL, LC) struct node *node_##LC(struct node *lhs);
#define DECL_BINOP(UL, LC)                                                     \
  struct node *node_##LC(struct node *lhs, struct node *rhs);

NODE_TYPES(DECL_LIT, DECL_UNOP, DECL_BINOP)

#undef DECL_LIT
#undef DECL_UNOP
#undef DECL_BINOP

int node_mark(struct node *node, struct node **head, int count, int visited);
void node_free(struct node *head, int visited);
void node_zerograd(struct node *head, int visited);
void node_codegen(FILE *fp, char *decl_fmt, char *ref_fmt, struct node *node,
                  int visited);
void node_eval(struct node *node, int visited);
void node_grad(struct node *node, int visited);
