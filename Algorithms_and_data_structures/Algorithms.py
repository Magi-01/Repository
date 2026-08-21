import heapq
from collections import deque, Counter, defaultdict

class Utility():
    def __init__(self):
        pass

    def merge(self, left: list, right: list) -> list:
        n = len(left)
        m = len(right)

        i,j = 0,0
        array = []

        while i < n and j < m:
            if left[i] < right[j]:
                array.append(left[i])
            else:
                array.append(right[j])

        while i < n:
            array.append(left[i])

        while j < n:
            array.append(right[j])

        return array

    def pivot(self, left: int, right: int, subarray: list) -> int:
        i = left-1

        for j in range(left, right):
            if subarray[j] < subarray[left]:
                i += 1
                subarray[i], subarray[j] = subarray[j], subarray[i]
                
        i += 1
        subarray[i], subarray[right] = subarray[right], subarray[i]

        return i

    def heapify(self, heap: list) -> list:
        return heap

    def collision_check(self, val):
        pass

#-----------------------------------------
# SORTING ALGORITHMS
#-----------------------------------------

class Sorting(Utility):
    def __init__(self):
        pass

    def bubble_sort(self, array: list) -> list:
        n = len(array)

        for i in range(n):
            for j in range(i,n-1):
                if array[j] > array[j+1]:
                    array[j], array[j+1] = array[j+1], array[j]
        return array

    def insertion_sort(self, array: list) -> list:
        n = len(array)
        i, j = 0, 1

        while i <= j:
            if array[i] < array[j]:
                array[i], array[j] = array[j], array[i]
                i = 0
            else:
                j += 1

            if i == j and j<n:
                i = 0
                j += 1

            i += 1

        return array

    def selection_sort(self, array: list) -> list:
        n = len(array)

        for i in range(n):
            for j in range(i):
                if array[i] < array[j]:
                    array[j], array[i] = array[i], array[j]

        return array

    def merge_sort(self, array: list) -> list:
        n = len(array)

        if n == 1:
            return array
        
        mid = n//2

        left = self.merge_sort(array[:mid])
        right = self.merge_sort(array[mid:])

        self.merge(left, right)

        return array

    def quick_sort(self, left: int, right: int, array: list) -> list:
        n = len(array)

        if n == 1:
            return array
        
        pi = self.pivot(left, right, array)
        self.quick_sort(left, pi-1, array)
        self.quick_sort(pi+1, right, array)
        return array

    def max_heap(self, array: list) -> list:
        return array

    def min_heap(self, array: list) -> list:
        return array

    def counting_sort(self, array: list) -> list:
        return array

    def radix_sort(self, array: list) -> list:
        return array

    def bucket_sort(self, array: list) -> list:
        return array

#-----------------------------------------
# DATA STORAGE TYPES
#-----------------------------------------

class LinkedList(Utility):
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

class HashTable(Utility):
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def hashing_function(self, value):
        pass


class Tree(Utility):
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

#-----------------------------------------
# SEARCHING ALGORITHMS
#-----------------------------------------

class TreeBST(Utility):
    # Tree version
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, tree: Tree):
        pass

    def remove(self, tree: Tree):
        pass

    def search(self, val, tree: Tree):
        pass

class BST(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

class BalancedBST(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

class AVLTree(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

class REDBLACKTree(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

class SPLAYTree(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

class BTree(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

class Tries(Utility):
    def __init__(self):
        pass

    def construct(self, array: list):
        pass

    def add(self, val, array: list):
        pass

    def remove(self, val, array: list):
        pass

    def search(self, val, array: list):
        pass

#-----------------------------------------
# GRAPHS
#-----------------------------------------

class Graph(Utility):
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

class GraphTraversal(Utility):
    def __init__(self):
        pass

    def dfs(self, graph: Graph):
        pass

    def bfs(self, graph: Graph):
        pass

    def dijkstra(self, graph: Graph):
        pass

    def Bellman_Ford(self, graph: Graph):
        pass

    def Floyd_Warshall(self, graph: Graph):
        # ALL PAIRS SHPRTEST PATH
        pass

    def Topological(self, graph: Graph):
        pass

    def Tarjan(self, graph: Graph):
        pass

    def Prim_Kruskal(self, graph: Graph):
        pass

#-----------------------------------------
# Common Dynamic Programming Problems
#-----------------------------------------

class DynamicProgramming(Utility):
    def __init__(self):
        pass

    def knapsack(self, Cost: int, size: list, pocket: list):
        pass

    def knapsack_repetition(self, Cost: int, Size: list, pocket: list):
        pass

    def minimum_coin(self, Cost: int, pocket: list):
        pass

    def minimum_coin_repetition(self, Cost: int, pocket: list):
        pass

    def maximum_coin(self, Cost: int, pocket: list):
        pass

    def maximum_coin_repetition(self, Cost: int, pocket: list):
        pass

    def longest_common_subsequence(self, text1: str, text2: str):
        pass

    def max_path_sum(self, graph: Graph):
        pass

    def equal_sum_subset(self, array: list):
        pass

    def palindrome(self, text: str):
        pass