from exercises import compression_sort, descending_bubble_sort, mean_pivot_quicksort, three_way_merge_sort

def test_compression_sort():
    tests = [
        (compression_sort([4, 2, 4, 1, 3, 2]), [1, 2, 3, 4]),
        (compression_sort([5, 5, 5]), [5]),
        (compression_sort([1, 2, 3]), [1, 2, 3]),
        (compression_sort([]), []),
        (compression_sort([9, 7, 8, 7, 9, 10]), [7, 8, 9, 10]),
        (compression_sort([-1, -3, -2, -1]), [-3, -2, -1]),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")
    # The original list must not be modified by the function
    data = [3, 1, 2, 3]
    if compression_sort(data) == [1, 2, 3] and data == [3, 1, 2, 3]:
        print("Test 6 passed.")
    else:
        print(f"Test 6 failed: input list mutated to {data}")


def test_descending_bubble_sort():
    tests = [
        ([3, 1, 4, 1, 5], [5, 4, 3, 1, 1]),
        ([1, 2, 3], [3, 2, 1]),
        ([5, 4, 3], [5, 4, 3]),
        ([], []),
        ([7], [7]),
        ([2, 2, 1, 3, 3], [3, 3, 2, 2, 1]),
    ]
    for i, (data, expected) in enumerate(tests):
        descending_bubble_sort(data) # The sort is in-place: check the mutated list
        if data != expected:
            print(f"Test {i} failed: got {data}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_three_way_merge_sort():
    tests = [
        (three_way_merge_sort([3, 1, 4, 1, 5, 9, 2, 6, 5]), [1, 1, 2, 3, 4, 5, 5, 6, 9]),
        (three_way_merge_sort([10, 9, 8, 7, 6, 5]), [5, 6, 7, 8, 9, 10]),
        (three_way_merge_sort([1, 2]), [1, 2]),
        (three_way_merge_sort([]), []),
        (three_way_merge_sort([4]), [4]),
        (three_way_merge_sort([-2, 5, -8, 0]), [-8, -2, 0, 5]),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_mean_pivot_quicksort():
    tests = [
        ([3, 1, 4, 1, 5, 9], [1, 1, 3, 4, 5, 9]),
        ([10, 9, 8, 7, 6], [6, 7, 8, 9, 10]),
        ([1, 2], [1, 2]),
        ([], []),
        ([7, 7, 7, 7], [7, 7, 7, 7]),
        ([5, -1, 3, -2, 4], [-2, -1, 3, 4, 5]),
    ]
    for i, (data, expected) in enumerate(tests):
        mean_pivot_quicksort(data) # The sort is in-place: check the mutated list
        if data != expected:
            print(f"Test {i} failed: got {data}, expected {expected}")
        else:
            print(f"Test {i} passed.")

def main():
    print("Running compression sort tests...")
    test_compression_sort()
    print("\nRunning descending bubble sort tests...")
    test_descending_bubble_sort()
    print("\nRunning 3-way merge sort tests...")
    test_three_way_merge_sort()
    print("\nRunning mean-pivot quicksort tests...")
    test_mean_pivot_quicksort()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()