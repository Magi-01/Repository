def compression_sort(A: list[int]) -> list[int]:
    n = len(A)
    i, j = 0, 1

    while i < n-1:
        if A[i] > A[j]:
            A[i], A[j] = A[j], A[i]
        if A[i] == A[j]:
            A.remove(A[i])
        if j == n:
            i += 1
            j = i
        j += 1

    return A  

def descending_bubble_sort(A: list[int]) -> list[int]:
    n = len(A)
    swap = False

    for i in range(n):
        for j in range(n):
            if A[i] < A[j]:
                A[i], A[j] = A[j], A[i]
                swap = True
        if swap:
            break

    return A
        

def three_way_merge_sort(A: list[int]) -> list[int]:
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return A
    if n == 2:
        if A[0] > A[1]:
            A[0], A[1] = A[1], A[0]
        return A
    def merge(left, middle, right):
        n = len(left)
        m = len(right)
        k = len(right)
        B = []
        i,j,l = 0,0,0
        while i < n and j < m and l < k:
            if left[i] > middle[j]:
                left[i], middle[j] = middle[j], left[i]
            if middle[i] > right[l]:
                middle[i], right[l] = right[l], middle[j]
            if left[i] > right[l]:
                left[i], right[l] = right[l], left[i]
        B = left+middle+right
        while i < n:
            B.append(left[i])
            i+=1
        while j < m:
            B.append(middle[j])
            j+=1
        while l < k:
            B.append(right[l])
            l+=1
        return B

    def merge_sort(A, n):
        if n < 1:
            return []
        if n == 1:
            return A[n-1]
        k1 = n//3
        k2 = 2*n//3
        left = merge_sort(A[0:k1])
        middle = merge_sort(A[k1+1:k2])
        right = merge_sort(A[k2+1:-1])
        return merge(left,middle,right)

    return merge_sort(A,n)

def mean_pivot_quicksort(A: list[int]) -> list[int]:
    def mean(A):
        tot = [ele/len(A) for ele in A]
        return sum(tot)
    pivot = mean(A)
    n = len(A)

    def quicksort(A, pivot):
        for i in range(n):
            if A[i] > pivot:
                pass
        
            elif A[i] < pivot:
                pass

    def quick(A, pivot):
        if len(A) < 1:
            return []
        if len(A) == 1:
            return A
        quicksort(A, pivot)
        return quick(A,pivot)
            

