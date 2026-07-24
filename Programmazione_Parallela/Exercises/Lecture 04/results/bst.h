#ifndef _BST_H
#define _BST_H

#include "tnode.h"

struct _bst {
  t_node root;
};

typedef struct _bst *bst;

bst make_bst (void);

void delete_bst (bst t);

void bst_insert (bst t, t_node n);

int bst_depth (bst t);

#endif