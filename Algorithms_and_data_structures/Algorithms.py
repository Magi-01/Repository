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

    def heapify(self, array: list, n: int, i: int) -> list:
        largest = i
        l = 2*i + 1
        r = 2*i + 2

        if l < n and array[l] > array[largest]:
            largest = l

        if r < n and array[r] > array[largest]:
            largest = i

        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            self.heapify(array, len(array), largest)
        return array

    def max_heap(self, array: list) -> list:
        # Used in Graph algorithms Dijkistra, Prim and Kruksal
        for i in range(len(array)//2 - 1, -1, -1):
            self.heapify(array, len(array), i)

        return array

    def collision_check(self, val):
        pass

#-----------------------------------------
# SORTING ALGORITHMS
#-----------------------------------------

class Sorting(Utility):
    def __init__(self):
        pass

    def bubble_sort(self, array: list):
        n = len(array)

        for i in range(n):
            for j in range(i,n-1):
                if array[j] > array[j+1]:
                    array[j], array[j+1] = array[j+1], array[j]

    def insertion_sort(self, array: list):
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

    def selection_sort(self, array: list):
        n = len(array)

        for i in range(n):
            for j in range(i):
                if array[i] < array[j]:
                    array[j], array[i] = array[i], array[j]

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

    def heap_sort(self, array: list) -> list:
        n = len(array)

        for i in range(n//2-1, -1, -1):
            self.heapify(array, len(array), i)
        for i in range(n-1, 0, -1):
            array[0], array[i] = array[i], array[0]
            self.heapify(array, i, 0)
        return array

    def counting_sort(self, array: list, exp1 = 1, radix = False) -> list:
        n = len(array)
        ans = [0]*n
        
        if radix:
            bucket = [0]*(10)

            for i in range(n):
                idx = array[i] // exp1
                bucket[idx%10] += 1

            for i in range(1, 10):
                bucket[i] += bucket[i-1]

            i = n - 1

            while i >= 0:
                idx = array[i] // exp1
                ans[bucket[idx%10] - 1] = array[i]
                bucket[idx%10] -= 1
                i -= 1

        else:
            bucket = [0]*(max(array) + 1)

            for i in range(n):
                bucket[array[i]] += 1

            for i in range(1, max(array) + 1):
                bucket[i] += bucket[i-1]

            for i in range(n - 1, -1, -1):
                temp = array[i]
                ans[bucket[temp] - 1] = temp
                bucket[temp] -= 1

        return ans

    def radix_sort(self, array: list) -> list:
        large = max(array)
        exp = 1
        while large / exp >= 1:
            self.counting_sort(array, exp, radix=True)
            exp *= 10
        return array

    def bucket_sort(self, array: list):
        # Space is O(n + k)
        buckets = [[]*len(array)]

        for num in array:
            buckets[int(float(len(array)*num))].append(num)

        for bucket in buckets:
            # Has worst case of O(n^2) if all elements fall within one bucket
            # in which case it it better to use divide and conquer methods

            # Has best case of O(n + k) where each bucket is equally 
            # distributed and, as such insertion sort runs on an array of size 
            # bucket
            self.insertion_sort(bucket)

        index = 0
        for bucket in buckets:
            for num in bucket:
                array[index] = num
                index += 1

#-----------------------------------------
# DATA STORAGE TYPES
#-----------------------------------------

class LinkedList(Utility):
    def __init__(self, val):
        self.val = val
        self.next = None

class HashTable(Utility):
    """
    Choose between various methods being:
    - chaining: Linked List
    - chaining: python List (dynamic arrays)
    - chaining: Self Balancing BST (AVL)
    - chaining: Self Balancing BST (Red-Black)
    - open addressing: Linear Probing
    - open addressing: Quadratic Probing
    - open addressing: Double Hashing

    """
    def __init__(self, val, method = "chaining: Linked List"):
        self.val = val
        self.position = self.hashing_function(val, method)

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def hashing_function(self, val, method):
        if method == "chaining: Linked List":
            bucket = val%10
            lst = {}
            lst[bucket] = LinkedList(val)
            while lst[bucket].next != None and lst[bucket].next.next != None:
                lst[bucket].val = lst[bucket].next
            lst[bucket].next = None # TO BE CONTINUED
            self.list = lst
            
        return val%10


class Tree(Utility):
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

# Call either max-heap or heapsort
class Heap(Utility):
    def __init__(self):
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