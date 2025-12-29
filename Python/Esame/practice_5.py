def bin_search_tree_keys_complete(t):
    if t == None:
        return True

    if (t.left == None and t.right != None) or (t.left != None and t.right == None):
        return False
    
    if t.left == None and t.right == None:
        return True
        
    return bin_search_tree_keys_complete(t.left) and bin_search_tree_keys_complete(t.right)