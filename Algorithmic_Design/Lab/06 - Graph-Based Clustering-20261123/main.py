import csv
from pathlib import Path

from exercises import customer_segmentation


def normalize_groups(groups: list[tuple[set[str], int]]) -> set[frozenset[str]]:
    return {frozenset(members) for members, _ in groups}


def test_pdf_example() -> bool:
    profiles = [
        [1.0, 2.0],
        [1.5, 1.8],
        [2.0, 2.5],
        [10.0, 50.0],
        [12.0, 52.0],
    ]
    labels = ["User_A", "User_B", "User_C", "User_D", "User_E"]
    expected = [
        ({"User_A", "User_B", "User_C"}, 3),
        ({"User_D", "User_E"}, 2),
    ]
    result = customer_segmentation(profiles, labels)
    passed = normalize_groups(result) == normalize_groups(expected)
    if passed:
        print("PDF example passed.")
    else:
        print(f"PDF example failed: got {result}, expected {expected}")
    return passed


def test_wholesale_csv() -> bool:
    csv_path = Path(__file__).with_name("wholesale_customers.csv")
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        feature_names = reader.fieldnames
        if feature_names is None:
            print("CSV test failed: the file has no header.")
            return False
        rows = list(reader)

    raw_profiles = [[float(row[name]) for name in feature_names] for row in rows]
    labels = [f"Customer_{index + 1}" for index in range(len(raw_profiles))]
    result = customer_segmentation(raw_profiles, labels)
    values_are_integers = all(
        value.is_integer() for profile in raw_profiles for value in profile
    )

    # Check that the groups contain every customer once, without assuming a fixed row order
    all_members = set()
    valid_groups = len(result) == 2
    for members, size in result:
        valid_groups = valid_groups and len(members) == size
        valid_groups = valid_groups and size * 3 >= len(labels)
        valid_groups = valid_groups and all_members.isdisjoint(members)
        all_members.update(members)
    passed = (
        values_are_integers
        and valid_groups
        and all_members == set(labels)
        and sum(size for _, size in result) == len(labels)
    )
    if passed:
        sizes = sorted(size for _, size in result)
        print(
            f"CSV test passed: {len(labels)} integer profiles split into "
            f"two groups of {sizes}."
        )
    else:
        print("CSV test failed: groups are incomplete, unbalanced, or contain non-integer data.")
        print(f"Loaded {len(labels)} customers; returned group sizes {[size for _, size in result]}.")
    return passed


def main() -> None:
    print("Running customer segmentation tests...")
    results = [test_pdf_example(), test_wholesale_csv()]
    if all(results):
        print("\nAll tests passed.")
    else:
        print("\nSome tests failed.")


if __name__ == "__main__":
    main()
