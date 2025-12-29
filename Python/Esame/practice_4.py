def bin_search_tree_add(t):
    if t == None:
        return None
    
    if t.left == None and t.right == None:
        return t.values
    
    return t.val + bin_search_tree_add(t.left) + bin_search_tree_add(t.right)