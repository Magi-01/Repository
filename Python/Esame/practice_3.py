def bin_search_tree_keys_in_interval(t,a,b):
    add = 0
    if t == None:
        return 0

    if t.key > b or t.key < a:
        return 0
    else:
        add = add + bin_search_tree_keys_in_interval(t.left,a,b)
        add = add + bin_search_tree_keys_in_interval(t.right,a,b)
    
    return add