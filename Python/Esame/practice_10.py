import matplotlib.pyplot as plt
import numpy as np

# Algorithms and their complexities
algo_data = {
    'Insertion Sort': {'avg_time': 'O(n^2)', 'worst_time': 'O(n^2)', 'space': 'O(1)'},
    'Binary Search': {'avg_time': 'O(log n)', 'worst_time': 'O(log n)', 'space': 'O(1)'},
    'Selection Sort': {'avg_time': 'O(n^2)', 'worst_time': 'O(n^2)', 'space': 'O(1)'},
    'Bubble Sort': {'avg_time': 'O(n^2)', 'worst_time': 'O(n^2)', 'space': 'O(1)'},
    'Merge Sort': {'avg_time': 'O(n log n)', 'worst_time': 'O(n log n)', 'space': 'O(n)'},
    'Heap Sort': {'avg_time': 'O(n log n)', 'worst_time': 'O(n log n)', 'space': 'O(1)'},
    'Quick Sort': {'avg_time': 'O(n log n)', 'worst_time': 'O(n^2)', 'space': 'O(log n)'},
    'Counting Sort': {'avg_time': 'O(n+k)', 'worst_time': 'O(n+k)', 'space': 'O(k)'},
    'Radix Sort': {'avg_time': 'O(nk)', 'worst_time': 'O(nk)', 'space': 'O(n+k)'},
    'Bucket Sort': {'avg_time': 'O(n+k)', 'worst_time': 'O(n^2)', 'space': 'O(n)'},
    'Quick Select': {'avg_time': 'O(n)', 'worst_time': 'O(n^2)', 'space': 'O(1)'},
    'Medians of Medians': {'avg_time': 'O(n)', 'worst_time': 'O(n)', 'space': 'O(1)'}
}

# Data structures and their complexities
data_struct_data = {
    'Linked List': {'access': 'O(n)', 'search': 'O(n)', 'insertion': 'O(1)', 'deletion': 'O(1)'},
    'Doubly Linked List': {'access': 'O(n)', 'search': 'O(n)', 'insertion': 'O(1)', 'deletion': 'O(1)'},
    'Array': {'access': 'O(1)', 'search': 'O(n)', 'insertion': 'O(n)', 'deletion': 'O(n)'},
    'Circular Array': {'access': 'O(1)', 'search': 'O(n)', 'insertion': 'O(n)', 'deletion': 'O(n)'},
    'Dictionary': {'search': 'O(1)', 'insertion': 'O(1)', 'deletion': 'O(1)'},
    'Stack': {'access': 'O(n)', 'search': 'O(n)', 'insertion': 'O(1)', 'deletion': 'O(1)'},
    'Queue': {'access': 'O(n)', 'search': 'O(n)', 'insertion': 'O(1)', 'deletion': 'O(1)'},
    'Splay Tree': {'access': 'O(log n)', 'search': 'O(log n)', 'insertion': 'O(log n)', 'deletion': 'O(log n)'},
    'Binary Search Tree': {'access': 'O(log n)', 'search': 'O(log n)', 'insertion': 'O(log n)', 'deletion': 'O(log n)'},
    'Hash Table': {'search': 'O(1)', 'insertion': 'O(1)', 'deletion': 'O(1)'},
}

# Specific algorithms and their complexities continued
specific_algo_data = {
    'DFS (Depth-First Search)': {'time': 'O(V+E)', 'space': 'O(V)'},
    'Tarjan\'s Algorithm': {'time': 'O(V+E)', 'space': 'O(V+E)'},
    'AVL Tree': {'access': 'O(log n)', 'search': 'O(log n)', 'insertion': 'O(log n)', 'deletion': 'O(log n)'},
    'Red-Black Tree': {'access': 'O(log n)', 'search': 'O(log n)', 'insertion': 'O(log n)', 'deletion': 'O(log n)'},
    'Topological Sort': {'time': 'O(V+E)', 'space': 'O(V)'},
    'Bellman-Ford Algorithm': {'time': 'O(VE)', 'space': 'O(V)'},
    'Dijkstra\'s Algorithm': {'time': 'O((V+E) log V)', 'space': 'O(V)'},
    'Floyd-Warshall Algorithm': {'time': 'O(V^3)', 'space': 'O(V^2)'},
    'Fibonacci Heap': {'insertion': 'O(1)', 'decrease_key': 'O(1)', 'merge': 'O(1)', 'extract_min': 'O(log n)', 'delete': 'O(log n)'}
}

# Plotting
fig, ax = plt.subplots(figsize=(18, 10))

# Color coding for complexities
color_map = {
    'O(1)': 'green',
    'O(log n)': 'blue',
    'O(n)': 'orange',
    'O(n+k)': 'yellow',
    'O(nk)': 'pink',
    'O(n log n)': 'lightblue',
    'O(n^2)': 'red',
    'O(nm)': 'purple',
    'O((V+E) log V)': 'brown',
    'O(V+E)': 'grey',
    'O(VE)': 'cyan',
    'O(V^3)': 'magenta',
    'O(k)': 'darkgreen',
    'O(V)': 'darkblue',
    'O(V^2)': 'darkred'   # This is the new line to add
}


# Create a scatter plot for each algorithm complexity
for i, (algo, complexities) in enumerate(algo_data.items()):
    ax.scatter(i, 3, s=1000, color=color_map[complexities['avg_time']], label=f"{algo}: Avg Time {complexities['avg_time']}")
    ax.scatter(i, 2, s=1000, color=color_map[complexities['worst_time']], label=f"{algo}: Worst Time {complexities['worst_time']}")
    ax.scatter(i, 1, s=1000, color=color_map[complexities['space']], label=f"{algo}: Space {complexities['space']}")

# Create a scatter plot for each data structure complexity
for i, (data_struct, complexities) in enumerate(data_struct_data.items()):
    if 'access' in complexities:
        ax.scatter(i+10, 3, s=1000, color=color_map[complexities['access']], label=f"{data_struct}: Access {complexities['access']}")
    ax.scatter(i+10, 2, s=1000, color=color_map[complexities['search']], label=f"{data_struct}: Search {complexities['search']}")
    ax.scatter(i+10, 1, s=1000, color=color_map[complexities['insertion']], label=f"{data_struct}: Insertion {complexities['insertion']}")

# Create a scatter plot for each specific algorithm complexity
for i, (specific_algo, complexities) in enumerate(specific_algo_data.items()):
    if 'time' in complexities:
        ax.scatter(i+20, 2.5, s=1000, color=color_map[complexities['time']], label=f"{specific_algo}: Time {complexities['time']}")
    if 'space' in complexities:
        ax.scatter(i+20, 1.5, s=1000, color=color_map[complexities['space']], label=f"{specific_algo}: Space {complexities['space']}")

# Formatting
ax.set_xticks([])
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Space Complexity', 'Worst Time Complexity', 'Average Time Complexity'])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='x-small')

plt.show()