def compression_sort(A: list[int]) -> list[int]:
    i = 0
    for value in A:
        # Scan the sorted result to find the insertion position of value
        pos = 0
        while pos < i and A[pos] < value:
            pos += 1
        # If value is already present at that position, skip it:
        # in this way every duplicate is removed while sorting
        if pos == i or A[pos] != value:
            A = A[:pos] + [value] + A[pos:i]
            i += 1
    return A
    # Each of the n elements is inserted into a sorted list that grows up to
    # n positions, so scanning and shifting cost 1 + 2 + ... + n in total
    # Time complexity (worst case): $\Theta(n^2)$


def swap(A: list[int], i: int, j: int) -> None:
    temp = A[i]
    A[i] = A[j]
    A[j] = temp


def descending_bubble_sort(A: list[int]) -> list[int]:
    n = len(A)
    for end in range(n - 1, 0, -1):
        swapped = False # Tracks the swaps made during this pass
        for i in range(end):
            if A[i] < A[i + 1]:
                # Adjacent elements are out of order for the descending sort:
                # swapping them makes the smallest element of the current
                # prefix bubble up towards position end
                swap(A, i, i + 1)

                swapped = True
        if not swapped:
            # Early stopping: a complete pass without swaps means that the
            # list is already sorted, so the remaining passes are skipped
            return A
    # In the worst case (ascending input) the early stopping never triggers
    # and the passes perform (n - 1) + (n - 2) + ... + 1 comparisons
    # Time complexity (worst case): $\Theta(n^2)$
    return A


def merge_three(left: list[int], middle: list[int], right: list[int]) -> list[int]:
    merged = []
    i = j = k = 0 # Current reading position inside each of the three sorted parts
    while i < len(left) or j < len(middle) or k < len(right):
        # Take the smallest among the current heads of the parts that are not
        # exhausted yet, so that the merged list stays sorted
        if (i < len(left)
                and (j >= len(middle) or left[i] <= middle[j])
                and (k >= len(right) or left[i] <= right[k])):
            merged.append(left[i])
            i += 1
        elif j < len(middle) and (k >= len(right) or middle[j] <= right[k]):
            merged.append(middle[j])
            j += 1
        else:
            merged.append(right[k])
            k += 1
    return merged
    # Every element of the three parts is appended exactly once with O(1)
    # comparisons, so with n = len(left) + len(middle) + len(right) the merge
    # is linear in the total number of elements
    # Time complexity (worst case): $O(n)$


def three_way_merge_sort(A: list[int]) -> list[int]:
    if len(A) <= 1: # Base case: empty list or single element
        return A[:] # Already sorted, return a copy

    # Split the list into three approximately equal parts
    first = len(A) // 3
    second = 2 * len(A) // 3
    left = three_way_merge_sort(A[:first])
    middle = three_way_merge_sort(A[first:second])
    right = three_way_merge_sort(A[second:])

    # Merge the three sorted parts into a single sorted list
    return merge_three(left, middle, right)
    # Three recursive calls on parts of size n / 3 plus the linear merge:
    # T(n) = 3T(n / 3) + O(n), which solves to Θ(n log n) with the master theorem
    # Time complexity (worst case): $\Theta(n \log n)$


def mean_pivot_index(A: list[int], low: int, high: int) -> int:
    # Arithmetic mean of the elements in the current sub-array
    mean = sum(A[low:high + 1]) / (high - low + 1)
    closest = low
    for i in range(low + 1, high + 1):
        # Keep the element whose value is closest to the mean; the strict
        # inequality keeps the first occurrence in case of ties
        if abs(A[i] - mean) < abs(A[closest] - mean):
            closest = i
    return closest
    # Two linear scans over the m = high - low + 1 elements of the sub-array:
    # one to compute the mean and one to find the closest element
    # Time complexity (worst case): $O(m)$


def lomuto_partition(A: list[int], low: int, high: int) -> int:
    pivot = A[high]
    store = low
    for i in range(low, high):
        if A[i] <= pivot:
            swap(A, i, store)
            store += 1
    swap(A, store, high)
    return store


def mean_pivot_quicksort_helper(A: list[int], low: int, high: int) -> None:
    if low >= high: # Base case: sub-array with zero or one element
        return

    # Move the pivot (the element closest to the mean) to the end of the sub-array
    pivot_index = mean_pivot_index(A, low, high)
    swap(A, pivot_index, high)

    # Lomuto partition: elements <= pivot go to its left, the others to its right
    store = lomuto_partition(A, low, high)

    # The pivot takes its final position (already done in lomuto_partition)

    # Sort recursively the two sub-arrays on the sides of the pivot
    mean_pivot_quicksort_helper(A, low, store - 1)
    mean_pivot_quicksort_helper(A, store + 1, high)
    # Each call costs O(m) for mean, pivot selection and partition on the
    # m = high - low + 1 elements.
    # Balanced splits makes 2 recursive calls on n/2 elements: T(n) = 2T(n / 2) + O(n).
    # That is Θ(n log n) with the master theorem.
    # But maximally unbalanced ones (e.g. all equal values) give T(n) = T(n - 1) + O(n)
    # Time complexity (average case): $\Theta(n \log n)$
    # Time complexity (worst case): $\Theta(n^2)$


def mean_pivot_quicksort(A: list[int]) -> list[int]:
    # Sort the whole list in-place recursing on the interval [0, n - 1]
    mean_pivot_quicksort_helper(A, 0, len(A) - 1)
    # Time complexity (average case): $\Theta(n \log n)$
    # Time complexity (worst case): $\Theta(n^2)$
    return A

