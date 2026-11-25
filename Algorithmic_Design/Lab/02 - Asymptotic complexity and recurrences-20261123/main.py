from exercises import count_equal_pairs, first_one, longest_increasing_run, prefix_sums

def test_longest_increasing_run():
    tests = [
        (longest_increasing_run([3, 5, 7, 2, 4, 6, 8, 1]), 4),
        (longest_increasing_run([1, 2, 2, 3]), 2),
        (longest_increasing_run([9, 7, 5]), 1),
        (longest_increasing_run([]), 0),
        (longest_increasing_run([1, 2, 3, 4, 5]), 5),
        (longest_increasing_run([5]), 1),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_count_equal_pairs():
    tests = [
        (count_equal_pairs([4, 1, 4, 4, 1]), 4),
        (count_equal_pairs([7, 7, 7]), 3),
        (count_equal_pairs([1, 2, 3]), 0),
        (count_equal_pairs([5]), 0),
        (count_equal_pairs([]), 0),
        (count_equal_pairs([2, 2, 2, 2]), 6),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_first_one():
    tests = [
        (first_one([0, 0, 0, 1, 1, 1]), 3),
        (first_one([1, 1, 1]), 0),
        (first_one([0, 0, 0]), -1),
        (first_one([0, 1]), 1),
        (first_one([]), -1),
        (first_one([0, 0, 0, 0, 1]), 4),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_prefix_sums():
    tests = [
        (prefix_sums([3, 1, 4, 2]), [3, 4, 8, 10]),
        (prefix_sums([2, -3, 5]), [2, -1, 4]),
        (prefix_sums([7]), [7]),
        (prefix_sums([]), []),
        (prefix_sums([1, 1, 1, 1, 1]), [1, 2, 3, 4, 5]),
        (prefix_sums([0, -1, 2, -3]), [0, -1, 1, -2]),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")

def main():
    print("Running longest increasing run tests...")
    test_longest_increasing_run()
    print("\nRunning count equal pairs tests...")
    test_count_equal_pairs()
    print("\nRunning first one tests...")
    test_first_one()
    print("\nRunning prefix sums tests...")
    test_prefix_sums()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()