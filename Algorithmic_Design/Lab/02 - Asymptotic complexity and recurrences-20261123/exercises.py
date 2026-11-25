def longest_increasing_run(A: list[int]) -> int:
    n = len(A)
    i, j = 0, 1
    long_index = 0
    while i < n and j < n:
        if A[j-1] >= A[j]:
            if long_index <= len(A[i:j]):
                long_index = len(A[i:j])
            i=j
        j+=1
    if long_index <= len(A[i:j]):
        long_index = len(A[i:j])
    return long_index



def count_equal_pairs(A: list[int]) -> int:
    n = len(A)
    if n < 2:
        return 0
    
    i, j = 0, 0
    pairs = set()

    for i in range(n):
        for j in range(i+1,n):
            if A[i] == A[j] and A[j]:
                pairs.add((i,j))
        

    return len(pairs)


def first_one(A: list[int]) -> int:
    n = len(A)
    if n < 1:
        return -1
    j = n//2 - 1
    i = 0

    while j > -1  and j < n-1:
        if A[j] >= A[j+1]:
            if A[j] == 1:
                n = j
                j = n//2 - 1
            elif A[j] == 0:
                i = j + 1
                j = (i+n)//2
        else:
            return j + 1
    if A[0] == 1:
        return 0
    else:
        return -1
            
            

def prefix_sums(A: list[int]) -> list[int]:
    n = len(A)
    if n == 1:
        return A

    for i in range(1,n):
        A[i] += A[i-1]

    return A