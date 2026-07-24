#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tnode.h"
#include "bst.h"
#include "print_tree.h"

void s_test (int n)
{
  int i = 0;
  bst t = make_bst();
  printf("Aggiunta dei nodi\n");
  for (;i < n; i++) {
    t_node n = make_t_node();
    n->key = i;
    bst_insert(t, n);
  }
  printf("L'albero e' profondo %d\n",bst_depth(t));
	print_bst(t);
  delete_bst(t);
}

void r_test (int n)
{
  int i = 0;
  bst t = make_bst();
  printf("Aggiunta dei nodi\n");
  for (;i < n; i++) {
    t_node n = make_t_node();
    n->key = rand() % 100;
    bst_insert(t, n);
  }
  printf("L'albero e' profondo %d\n",bst_depth(t));
	print_bst(t);
  delete_bst(t);
}

int main (int argc, char * argv[])
{
  int n = 0;
  srand((int)time(NULL));
  printf("Numero di nodi: ");
  scanf("%d",&n);
  s_test(n);
  r_test(n);
  return 0;
}