def dfs_visit(
    adj_matrix: list[list[int]], node: int, colour: list[str],
    parent: list[int | None], discover: list[int], finish: list[int], time: int
) -> int:
    # 2. We need to increment the global time and mark the discovery time of the current node.
    #    Then we mark the current node as discovered (Grey).
    time += 1
    discover[node] = time
    colour[node] = "Grey" 

    # 3. We iterate over all neighbors of the current node.
    #    If a neighbor is not discovered (White), we recursively visit it by calling DFS on that neighbor.
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] == 1 and colour[neighbor] == "White":
            parent[neighbor] = node
            time = dfs_visit(adj_matrix, neighbor, colour, parent, discover, finish, time)
    
    # 4. After visiting all neighbors, we mark the current node as fully explored (Black)
    #    and record its finishing time.
    colour[node] = "Black" # Black = node is fully explored
    time += 1
    finish[node] = time
    
    return time
    # Each of k vertices in the visited subtree scans one row of length n.
    # Time complexity (worst case): $O(kn)$


def dfs(adj_matrix: list[list[int]]) -> tuple[list[int | None], list[int], list[int]]:
    # We need 6 things:
    # + The number of vertices
    # + A colour array to track the state of each vertex (White, Grey, Black)
    # + A parent array to reconstruct the DFS forest
    # + A discovery time array
    # + A finishing time array
    # + A global time counter
    
    n = len(adj_matrix)
    colour = ["White"] * n
    parent = [None] * n # If the interpreter flags this like a type issue, you can ignore it. Or add 'list[int | None]' before '='.
    discover = [0] * n
    finish = [0] * n
    time = 0

    # 1. We iterate over all nodes and start a DFS visit for each undiscovered node.
    #    In this way, we ensure that even the non-connected components are discovered.
    for node in range(n):
        if colour[node] == "White":
            time = dfs_visit(adj_matrix, node, colour, parent, discover, finish, time)

    return parent, discover, finish
    # Each vertex is visited once and scans its complete row of n entries,
    # where n is the number of vertices in the adjacency matrix.
    # Time complexity (worst case): $O(n^2)$


def user_group_discovery(adj_matrix: list[list[int]], user_labels: list[str]) -> list[tuple[set[str], int]]:

    # 1. We execute DFS to determine the parent links of each user in the DFS forest.
    #    Discovery and finishing times are not needed for our case.
    parent, _, _ = dfs(adj_matrix)
    n = len(adj_matrix)

    # 2. We initialize the components array to store the users belonging to each connected component.
    components = []
    groups = []

    # 3. We need a set for each node, in this way if nobody is connected to another user, 
    #    they will still have their own group.  
    for node in range(n):
        components.append(set())

    # 4. At this point we follow the parents:
    #    The concept is to go through the parent links until we reach the root of the (sub)tree for each user.
    #    In this way, we can group all users that belong to the same connected component under the same root.
    for node in range(n):
        root = node
        parent_node = parent[root]
        while parent_node is not None:
            root = parent_node
            parent_node = parent[root]
        components[root].add(user_labels[node])

    # 5. Each root represents one connected component.
    #    And we want to consider only the non-empty components corresponding to the roots.
    for root in range(n):
        if parent[root] is None and components[root]: # If we find it
            component = components[root]
            groups.append((component, len(component))) # We store the component and its size
    return groups
    # DFS scans n rows of n entries; each of n parent chains has length at
    # most n, where n is the number of users.
    # Time complexity (worst case): $O(n^2)$


def initialize_single_source(n: int, start: int) -> tuple[list[float], list[int]]:
    # This function makes sure that the distance to the start node is zero and all other distances are set to infinity.
    distance = [float("inf")] * n
    parent = [-1] * n
    distance[start] = 0
    return distance, parent
    # Initializing two arrays of n locations takes linear time.
    # Time complexity (worst case): $O(n)$


def extract_min(distance: list[float], in_queue: list[bool]) -> int:
    minimum_node = -1 # Base case: no node has been found yet
    for node in range(len(distance)):
        # Skip nodes that are not in the queue and continue searching for the minimum among queued nodes.
        if in_queue[node] and (
            minimum_node == -1 or distance[node] < distance[minimum_node]
        ):
            minimum_node = node
    return minimum_node
    # Scan all n locations once to find the queued node with minimum distance.
    # Time complexity (worst case): $O(n)$


def relax(u: int, v: int, cost_matrix: list[list[float]],distance: list[float], parent: list[int]) -> None:
    # This is the formula for relaxing an edge.
    new_distance = distance[u] + cost_matrix[u][v]
    if new_distance < distance[v]:
        distance[v] = new_distance
        parent[v] = u
    # The comparison and possible updates use constant time.
    # Time complexity (worst case): $O(1)$


def dijkstra(cost_matrix: list[list[float]], start: int) -> tuple[list[float], list[int]]:
    # We need 5 things:
    # + The number of nodes in the graph (n)
    # + A distance array
    # + A parent array
    # + A queue to track which nodes still need to be processed
    # + A remaining count of queued nodes
    n = len(cost_matrix)
    distance, parent = initialize_single_source(n, start)
    in_queue = [True] * n
    remaining = n

    # Keep going while there are still nodes in the queue
    while remaining > 0:

        # 1. Extract the queued location with the smallest tentative distance.
        #    Remove that location from the in-queue and decrement the remaining location to visit.
        u = extract_min(distance, in_queue)
        in_queue[u] = False
        remaining -= 1

        # 2. If the smallest distance is infinity, all remaining nodes are unreachable.
        if distance[u] == float("inf"):
            break 

        # 3. Relax all neighbors of u that are still in the queue and have a finite edge cost.
        for v in range(n):
            if in_queue[v] and cost_matrix[u][v] != float("inf"):
                relax(u, v, cost_matrix, distance, parent)

    return distance, parent
    # At most n rounds each scan n locations for Extract-min and n entries
    # in the matrix row, so with n locations the total work is quadratic.
    # Time complexity (worst case): $O(n^2)$


def min_distribution_costs(cost_matrix: list[list[float]], node_labels: list[str], start_node: str) -> list[tuple[str, float]]:
    # Base case: if the start node is not in the list of node labels, return an empty list
    if start_node not in node_labels:
        return [] 

    # Get the index of the start node in the list of node labels
    # Execute Dijkstra's algorithm from the start node. We only need the distances.
    start = node_labels.index(start_node)
    dist, _ = dijkstra(cost_matrix, start)

    # Pair every location with its minimum cost; unreachable ones keep inf
    result = []
    for i in range(len(node_labels)):
        result.append((node_labels[i], dist[i]))
    return result
    # A single Dijkstra run dominates the final linear scan over the locations
    # Time complexity (worst case): $O(n^2)$
