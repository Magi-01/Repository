def max_continuos_sum(Q):
    if Q == None:
        return 0
    
    maxcount = 0
    count = 0
    i = 0

    while i in range(len(Q)):
        if Q[i+1] >= Q[i]:
            count += 1
        i += 1
        if count > maxcount:
            maxcount = count
            count = 0