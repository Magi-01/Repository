from exercises import build_max_heap, extract_max, search, string_radix_sort

def test_build_max_heap():
    tests = [
        ([4, 1, 3, 2, 16, 9, 10, 14, 8, 7], [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]),
        ([], []),
        ([5], [5]),
        ([1, 2], [2, 1]),
        ([3, 3, 3], [3, 3, 3]),
        ([1, 2, 3, 4, 5], [5, 4, 3, 1, 2]),
    ]
    for i, (data, expected) in enumerate(tests):
        build_max_heap(data, len(data)) # The heap is built in-place: check the mutated list
        if data != expected:
            print(f"Test {i} failed: got {data}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_extract_max():
    tests = [
        ([16, 14, 10, 8, 7, 9, 3, 2, 4, 1], 10, (16, 9), [14, 8, 10, 4, 7, 9, 3, 2, 1, 1]),
        ([], 0, None, []),
        ([42], 1, (42, 0), [42]),
        ([2, 1], 2, (2, 1), [1, 1]),
        ([5, 5, 3], 3, (5, 2), [5, 3, 3]),
        ([9, 8, 7, 6, 5], 5, (9, 4), [8, 6, 7, 5, 5]),
        ([14, 8, 10, 4, 7, 9, 3, 2, 1, 1], 9, (14, 8), [10, 8, 9, 4, 7, 1, 3, 2, 1, 1]),
    ]
    # The list keeps its length: the positions past the new heap size hold
    # stale values, so each test checks the tuple, the size and the whole list
    for i, (data, heap_size, expected_return, expected_list) in enumerate(tests):
        result = extract_max(data, heap_size)
        if result != expected_return or data != expected_list:
            print(f"Test {i} failed: got {result} and {data}, expected {expected_return} and {expected_list}")
        else:
            print(f"Test {i} passed.")


def test_search():
    heap = [14, 8, 10, 4, 7, 9, 3, 2, 1] # The heap obtained in the PDF examples
    tests = [
        (search(heap, 9), 5),
        (search(heap, 100), -1),
        (search(heap, 14), 0),
        (search(heap, 1), 8),
        (search([], 5), -1),
        (search([7, 7, 7], 7), 0),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_string_radix_sort():
    tests = [
        (string_radix_sort(["cat", "car", "apple", "bat"]), ["apple", "bat", "car", "cat"]),
        (string_radix_sort(["a", "ab", "abc", "b"]), ["a", "ab", "abc", "b"]),
        (string_radix_sort(["zebra", "ant"]), ["ant", "zebra"]),
        (string_radix_sort([]), []),
        (string_radix_sort(["b", "a", "c"]), ["a", "b", "c"]),
        (string_radix_sort(["aa", "aa", "ab"]), ["aa", "aa", "ab"]),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")

def main():
    print("Running build max heap tests...")
    test_build_max_heap()
    print("\nRunning extract max tests...")
    test_extract_max()
    print("\nRunning heap search tests...")
    test_search()
    print("\nRunning string radix sort tests...")
    test_string_radix_sort()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()