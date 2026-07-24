"""
A[[a_1 , b_2],...,[a_n , b_n]] with [a , b] meaning that
b is a must do exam for a
A is of size K
The algorithm returns True iff all exams can be done
meaning a cannot appear in a[1][i] and b cannot appear in a[0][i]
"""

def can_complete_course(exams):
    graph = {}
    visited = {}
    rec_stack = {}

    # Costruzione del grafo dag dai prerequisiti
    for a, b in exams:
        if b not in graph:
            graph[b] = []
        graph[b].append(a)

    # Funzione DFS per visitare il grafo e rilevare cicli
    def dfs(node):
        if node in rec_stack:
            return True  # C'è un ciclo
        if node in visited:
            return False  # Nodo già visitato, nessun ciclo
        visited[node] = True
        rec_stack[node] = True
        if node in graph:
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
        rec_stack.pop(node)
        return False

    # Visita ogni nodo del grafo
    for exam in graph.keys():
        if dfs(exam):
            return False  # Ciclo trovato, impossibile completare il corso di studi

    return True  # Nessun ciclo trovato, possibile completare il corso di studi


# Esempio di utilizzo:
exams_list = [['a', 'b'], ['c', 'd'], ['b', 'c']]
print(can_complete_course(exams_list))  # Output: True
