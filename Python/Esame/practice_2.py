def bin_search_tree_check(t,a,b):
    if t == None:
        return True

    if t.key > b or t.key < a:
        return False
    
    return bin_search_tree_check(t.left,a,t.key-1) and bin_search_tree_check(t.right,t.key+1,b)