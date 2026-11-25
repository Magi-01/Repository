def user_group_discovery(adj_matrix: list[list[int]], user_labels: list[str]) -> list[tuple[set[str], int]]:
    # We need 3 things:
    # + The size of the graph (number of users)
    # + A visited array to track which node has already been assigned to a group
    # + A list to store the discovered groups

    n = len(adj_matrix) # Can be obtained directly from the matrix or the user_labels list
    visited = [False] * n
    groups = []
    
    for start in range(n):
        if visited[start]: # If the user has already been assigned to a group
            continue # Skip it
        
        # Iterative DFS from start: collect every user reachable through
        # direct interactions
        component = set()
        stack = [start] # This stack keeps track of neighboring users to explore next
        visited[start] = True 

        while stack: 
            node = stack.pop() # Remove the next user
            component.add(user_labels[node]) # Add the user to the current group
            for neighbor in range(n):
                # If the neighbor is directly connected and not yet visited, 
                # add it to the stack
                # In this way each user is added to the stack at most once 
                # and visited exactly once
                if adj_matrix[node][neighbor] == 1 and not visited[neighbor]: 
                    visited[neighbor] = True
                    stack.append(neighbor)

        groups.append((component, len(component)))
    return groups
    # Every user is pushed and popped exactly once, and for each of them the
    # full matrix row of length n is scanned, with n = number of users
    # Time complexity (worst case): $O(n^2)$


def dijkstra(cost_matrix: list[list[float]], start: int) -> tuple[list[float], list[int]]:
    # We need 5 things:
    # + The number of locations
    # + A distance array to store the minimum cost from start to each location
    # + A predecessor array to reconstruct the cheapest route
    # + A visited array to track which locations have their minimum cost finalized
    # + The start location itself
    
    n = len(cost_matrix)
    dist = [float("inf")] * n 
    pred = [-1] * n
    visited = [False] * n
    dist[start] = 0

    # TThe objective is to extract the unvisited location with the smallest tentative cost:
    # with strictly positive costs that can no longer be improved
    for _ in range(n):
        u = -1 # Base initialization

        for i in range(n):
            # Consider location i only if it has not been visited yet
            # and it has the smallest tentative cost among unvisited locations so far
            if not visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i # Update the candidate to the next location 

        # If no unvisited location with a finite tentative cost is found, break the loop
        if u == -1 or dist[u] == float("inf"):
            break # The remaining locations are unreachable from the location
        visited[u] = True

        # Relax every direct route that leaves u; missing routes cost inf,
        # so they can never improve a tentative cost
        for v in range(n):
            if dist[u] + cost_matrix[u][v] < dist[v]:
                dist[v] = dist[u] + cost_matrix[u][v]
                pred[v] = u

    return dist, pred
    # n extraction rounds, each scanning n locations for the minimum and a
    # matrix row of length n for the relaxation, with n = len(cost_matrix)
    # Time complexity (worst case): $O(n^2)$


def min_distribution_costs(cost_matrix: list[list[float]], node_labels: list[str], start_node: str) -> list[tuple[str, float]]:
    if start_node not in node_labels:
        return [] # Degenerate case: empty network or unknown warehouse

    # Get the index of the start node in the list of node labels
    start = node_labels.index(start_node)
    dist, _ = dijkstra(cost_matrix, start)

    # Pair every location with its minimum cost; unreachable ones keep inf
    return [(node_labels[i], dist[i]) for i in range(len(node_labels))]
    # A single Dijkstra run dominates the final linear scan over the locations
    # Time complexity (worst case): $O(n^2)$
