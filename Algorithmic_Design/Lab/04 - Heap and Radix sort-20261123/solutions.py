def max_heapify(A: list[int], i: int, heap_size: int) -> None:
    # Restore the Max-Heap property in the subtree rooted at index i,
    # assuming that its left and right subtrees are already Max-Heaps
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    # The parent must be larger than both children; ignore the children
    # that fall outside the current heap size
    if left < heap_size and A[left] > A[largest]:
        largest = left
    if right < heap_size and A[right] > A[largest]:
        largest = right
    if largest != i:
        # Swap the parent with its largest child and keep sifting down,
        # because the swapped element may violate the property below
        temp = A[i]
        A[i] = A[largest]
        A[largest] = temp
        max_heapify(A, largest, heap_size)
    # At each call the index moves one level down the heap, so with
    # n = heap_size at most O(log n) calls are made, doing O(1) work each
    # Time complexity (worst case): $O(\log n)$


def build_max_heap(A: list[int], heap_size: int) -> None:
    # Bottom-up approach: leaves are already trivial Max-Heaps, so sift down
    # every internal node starting from the parent of the last leaf
    for i in range(heap_size // 2 - 1, -1, -1):
        max_heapify(A, i, heap_size)
    # n / 2 sift-down operations; summing the heights of all the nodes
    # (most subtrees are short) gives a tight linear bound
    # Time complexity (worst case): $O(n)$


def extract_max(A: list[int], heap_size: int) -> tuple[int, int] | None:
    if heap_size == 0: # Base case: the heap is empty
        return None
    maximum = A[0] # In a Max-Heap the root is always the maximum
    # Change the root with the last element and reduce the heap size
    A[0] = A[heap_size - 1]
    heap_size -= 1
    if heap_size > 0:
        # Restore the Max-Heap property sifting the new root down
        max_heapify(A, 0, heap_size)
    return maximum, heap_size
    # One sift-down over the height of the heap, with n = heap_size
    # Time complexity (worst case): $O(\log n)$


def heap_search(A: list[int], i: int, value: int) -> int:
    if i >= len(A) or A[i] < value:
        # Pruning: in a Max-Heap every descendant of i is <= A[i], so when
        # A[i] is already smaller than value the whole subtree is skipped
        return -1
    if A[i] == value:
        return i
    # Otherwise the value can only be in one of the two subtrees
    left = heap_search(A, 2 * i + 1, value)
    if left != -1:
        return left
    return heap_search(A, 2 * i + 2, value)
    # Each node is visited at most once, and whole subtrees are pruned;
    # with n = len(A) the visits remain linear in the worst case
    # Time complexity (worst case): $Theta(n)$


def search(A: list[int], value: int) -> int:
    # Start the pruned search from the root of the heap
    return heap_search(A, 0, value)
    # Time complexity (worst case): $Theta(n)$


def char_key(s: str, pos: int) -> int:
    if pos >= len(s):
        return 0 # Missing characters conceptually precede the letter 'a'
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for index in range(len(alphabet)):
        if s[pos] == alphabet[index]:
            return index + 1
    return 0 # Inputs are lowercase English strings, as specified by the exercise


def counting_sort_by_char(A: list[str], pos: int) -> list[str]:
    # Stable Counting Sort that orders the strings by the character at pos
    counts = [0] * 27 # Bucket 0 for missing characters, 1..26 for 'a'..'z'
    for s in A:
        counts[char_key(s, pos)] += 1
    # Exclusive prefix sums: counts[k] becomes the first free slot of bucket k
    start = 0
    for k in range(27):
        temp = counts[k]
        counts[k] = start
        start += temp
    result = [""] * len(A)
    for s in A: # Forward scan: equal keys keep their previous order (stability)
        key = char_key(s, pos)
        result[counts[key]] = s
        counts[key] += 1
    return result
    # One pass to count and one to place over n = len(A) strings; the
    # alphabet size is a constant (27 buckets)
    # Time complexity (worst case): $O(n)$


def string_radix_sort(A: list[str]) -> list[str]:
    if not A:
        return []
    # LSD approach: process the character positions from the rightmost one
    # of the longest string back to position 0
    max_len = max(len(s) for s in A)
    result = list(A)
    for pos in range(max_len - 1, -1, -1):
        result = counting_sort_by_char(result, pos)
    return result
    # d = max_len passes of a linear Counting Sort over n = len(A) strings;
    # the stability of each pass preserves the order fixed by the suffixes
    # Time complexity (worst case): $O(d \cdot n)$

