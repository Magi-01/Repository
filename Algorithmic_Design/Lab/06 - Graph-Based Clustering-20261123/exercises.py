from math import sqrt


def build_dissimilarity_graph(user_profiles: list[list[float]]) -> list[list[float]]:
    n = len(user_profiles)
    graph = []

    # Create an empty row for every customer. The diagonal stays zero because
    # a customer's distance from itself is zero.
    for _ in range(n):
        row = [0.0] * n
        graph.append(row)

    # Calculate each pair once, then store the distance in both directions.
    for first in range(n):
        for second in range(first + 1, n):
            squared_distance = 0.0

            # Compute the squared Euclidean distance between the two user profiles, for all features.
            for feature in range(len(user_profiles[first])):

                # Calculate the difference for this feature and add its square to the squared distance.
                difference = user_profiles[first][feature] - user_profiles[second][feature]
                squared_distance += difference * difference

            # At this point, squared_distance contains the sum of squared differences for all features.
            distance = sqrt(squared_distance)

            # Store the computed distance in the graph for both directions.
            # Assign the distance to both [first][second] and [second][first] in the graph.
            graph[first][second] = distance
            graph[second][first] = distance

    return graph


def customer_segmentation(
    user_profiles: list[list[float]], user_labels: list[str]
) -> list[tuple[set[str], int]]:

    #graph = build_dissimilarity_graph(user_profiles)
    pass