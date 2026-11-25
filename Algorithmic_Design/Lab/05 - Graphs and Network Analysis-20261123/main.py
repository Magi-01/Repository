from exercises import min_distribution_costs, user_group_discovery

def normalize_groups(groups: list[tuple[set[str], int]]) -> set:
    if groups is None:
        return set()
    # The order of the groups and of the users within each set is irrelevant,
    # so comparisons are made on frozensets of the members
    return set(frozenset(members) for members, _ in groups)


def test_user_group_discovery():
    matrix = [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    tests = [
        (user_group_discovery(matrix, ["A", "B", "C", "D", "E", "F"]), [({"A", "B", "C"}, 3), ({"D", "E"}, 2), ({"F"}, 1)]),
        (user_group_discovery([], []), []),
        (user_group_discovery([[0, 0, 0], [0, 0, 0], [0, 0, 0]], ["X", "Y", "Z"]), [({"X"}, 1), ({"Y"}, 1), ({"Z"}, 1)]),
        (user_group_discovery([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], ["A", "B", "C", "D"]), [({"A", "B", "C", "D"}, 4)]),
        (user_group_discovery([[0, 1, 1], [1, 0, 1], [1, 1, 0]], ["A", "B", "C"]), [({"A", "B", "C"}, 3)]),
        (user_group_discovery([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], ["A", "B", "C", "D"]), [({"A", "B"}, 2), ({"C", "D"}, 2)]),
    ]
    for i, (result, expected) in enumerate(tests):
        if normalize_groups(result) != normalize_groups(expected):
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")
    # The reported size must match the number of users in each set
    result = user_group_discovery(matrix, ["A", "B", "C", "D", "E", "F"])
    if result is None:
        print("Test 6 failed: result is None")
    elif all(len(members) == size for members, size in result):
        print("Test 6 passed.")
    else:
        print(f"Test 6 failed: sizes do not match the sets in {result}")


def test_min_distribution_costs():
    costs = [
        [0, 8, 3, float("inf"), float("inf")],
        [float("inf"), 0, float("inf"), 4, float("inf")],
        [float("inf"), 2, 0, 9, float("inf")],
        [float("inf"), float("inf"), float("inf"), 0, float("inf")],
        [float("inf"), float("inf"), float("inf"), float("inf"), 0],
    ]
    labels = ["A", "B", "C", "D", "E"]
    inf = float("inf")
    tests = [
        (min_distribution_costs(costs, labels, "A"), [("A", 0), ("B", 5), ("C", 3), ("D", 9), ("E", inf)]),
        (min_distribution_costs(costs, labels, "D"), [("A", inf), ("B", inf), ("C", inf), ("D", 0), ("E", inf)]),
        (min_distribution_costs([[0]], ["A"], "A"), [("A", 0)]),
        (min_distribution_costs([[0, inf], [inf, 0]], ["A", "B"], "A"), [("A", 0), ("B", inf)]),
        (min_distribution_costs([[0, 10, 2], [inf, 0, inf], [inf, 3, 0]], ["A", "B", "C"], "A"), [("A", 0), ("B", 5), ("C", 2)]),
        (min_distribution_costs([], [], "A"), []),
    ]
    for i, (result, expected) in enumerate(tests):
        if result != expected:
            print(f"Test {i} failed: got {result}, expected {expected}")
        else:
            print(f"Test {i} passed.")

def main():
    print("Running user group discovery tests...")
    test_user_group_discovery()
    print("\nRunning min distribution costs tests...")
    test_min_distribution_costs()
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()