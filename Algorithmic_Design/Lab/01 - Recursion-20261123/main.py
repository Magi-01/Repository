from exercises import annual_interest, fibonacci, power, subset_sum

def test_power():
    tests = [
        (power(2, 3), 8),
        (power(5, 0), 1),
        (power(3, 4), 81),
        (power(10, 1), 10),
        (power(0, 8), 0),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_annual_interest():
    tests = [
        (annual_interest(1000.0, 0.05, 2), 1102.5),
        (annual_interest(500.0, 0.10, 3), 665.5),
        (annual_interest(100.0, 0.0, 5), 100.0),
        (annual_interest(250.0, 0.10, 0), 250.0),
        (annual_interest(200.0, 1.0, 2), 800.0),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_fibonacci():
    tests = [
        (fibonacci(0), 0),
        (fibonacci(1), 1),
        (fibonacci(6), 8),
        (fibonacci(8), 21),
        (fibonacci(10), 55),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")


def test_subset_sum():
    tests = [
        (subset_sum([3, 34, 4, 12, 5, 2], 9), True),
        (subset_sum([3, 34, 4, 12, 5, 2], 30), False),
        (subset_sum([], 0), True),
        (subset_sum([1, 2, 3], 6), True),
        (subset_sum([1, 2, 3], 7), False),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")

def main():
    print("Running power tests...")
    test_power()
    print("\nRunning annual interest tests...")
    test_annual_interest()
    print("\nRunning fibonacci tests...")
    test_fibonacci()
    print("\nRunning subset sum tests...")
    test_subset_sum()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()