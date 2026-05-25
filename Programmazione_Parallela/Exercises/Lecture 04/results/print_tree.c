#include <stdio.h>
#include "tnode.h"
#include "bst.h"
#include "print_tree.h"

void print_node_cascade(t_node n)
{
  if (n == NULL) {
    printf(".");
  } else if (n->left == NULL && n->right == NULL) {
    printf("%d", n->key);
  } else {
    printf("(%d ", n->key);
    print_node_cascade(n->left);
    printf(" ");
    print_node_cascade(n->right);
    printf(")");
  }
}

void print_bst(bst t)
{
  if (t == NULL) {
    return;
  }
  print_node_cascade(t->root);
  printf("\n");
}