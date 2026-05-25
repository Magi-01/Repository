# A[1,...,n] and n>0
# If A is min-heap, then return True
# Else False


def min_heap_check(a):
    if len(a) < 2:
        return a
    i = 0
    while i < len(a) and pow(2,i)+1 < len(a):
        if a[i] > a[pow(2,i)] or a[i] > a[pow(2,i)+1]:
            return False
        i += 1
    return True

a = [10,2,3,4,5]
k = min_heap_check(a)
print(k)