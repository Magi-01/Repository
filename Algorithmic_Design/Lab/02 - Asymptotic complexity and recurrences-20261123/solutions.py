def longest_increasing_run(A: list[int]) -> int:
    if not A: # Base case: the list is empty
        return 0 # With no positions there is no run

    longest = 1 # Length of the longest run found so far
    current = 1 # Length of the run that ends at the current position
    for i in range(1, len(A)):
        if A[i] > A[i - 1]:
            # The run continues: extend it by one position
            current += 1
        else:
            # Equal or decreasing values break the run: a new one starts here
            current = 1
        if current > longest:
            longest = current
    return longest
    # One left-to-right scan: each element is compared once with its
    # predecessor and every update costs O(1), so the total work is linear
    # Time complexity (worst case): $O(n)$


def count_equal_pairs(A: list[int]) -> int:
    count = 0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            # Compare every pair of distinct positions (i, j) with i < j,
            # so that each pair is counted exactly once
            if A[i] == A[j]:
                count += 1
    return count
    # The outer scan runs n times and the inner one n - i - 1 times, hence
    # the comparisons are (n - 1) + (n - 2) + ... + 1 = n(n - 1) / 2
    # Time complexity (worst case): $O(n^2)$


def first_one_helper(A: list[int], low: int, high: int) -> int:
    if low > high: # Base case: the interval is empty
        return -1 # No 1 can occur in an empty interval

    mid = (low + high) // 2
    if A[mid] == 1 and (mid == 0 or A[mid - 1] == 0):
        return mid # A[mid] is a 1 and everything before it is a 0

    if A[mid] == 0:
        # The first 1, if it exists, lies strictly to the right of mid
        return first_one_helper(A, mid + 1, high)
    # A[mid] is a 1 but not the first one: keep searching to the left
    return first_one_helper(A, low, mid - 1)
    # Every call halves the interval [low, high], so with n = high - low + 1
    # the recurrence is T(n) = T(n / 2) + O(1).
    # Since the input is halved at each step, it takes at most O(log n) 
    # recursive calls to reach the base case, doing O(1) work per call.
    # Time complexity (worst case): $O(\log n)$


def first_one(A: list[int]) -> int:
    # Delegate the search to the helper on the whole index interval [0, n - 1]
    return first_one_helper(A, 0, len(A) - 1)
    # A single call to the helper on an interval of size n
    # Time complexity (worst case): $O(\log n)$


def prefix_sums(A: list[int]) -> list[int]:
    if len(A) <= 1: # Base case: empty list or single element
        return A[:] # Return a copy, so that A is never modified

    # Split the list into two approximately equal halves
    mid = len(A) // 2
    left = prefix_sums(A[:mid])
    right = prefix_sums(A[mid:])

    # left[-1] is the sum of the whole left half: every prefix sum of the
    # right half has to be shifted by that amount
    offset = left[-1]

    for i in range(len(right)):
        right[i] += offset

    return left + right
    # Two recursive calls on two halves plus linear work for slicing,
    # concatenation and the shift: T(n) = 2T(n / 2) + O(n).
    # The recursion tree is O(log n) levels deep, and the total work 
    # done across all nodes at any single level sums up to O(n).
    # Time complexity (worst case): $O(n \log n)$

