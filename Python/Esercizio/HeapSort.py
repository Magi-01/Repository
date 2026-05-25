#2.) 
# Per il zero livello ho d^0 in quanto un elemento genitore.
# Per il primo livello ho d^1 in quanto d figli
# Per il secondo livello ho d^2 in quanto d figli
# ...
# Per il k-esimo livello ho d^k figli
# La somma di tutti i nodi di ogni livello dev'essere uguale ad n:
# 1+d+d^2+d^3+...d^k = n con k l'altezza
# Qusto é la serie geometrica (d^(k+1)-1)/(d-1) = n
# Da cui si ricava d^(k+1) = n(d-1)+1 -> k+1 = log_d(n(d-1)+1)
# -> k = log_d(n(d-1)+1) - 1

#1.)
import random
import time
from matplotlib import pyplot as plt
import numpy
from tqdm import tqdm
class Heap():
    def __init__(self, d):
        self.d = d

    def MAXheapify(self, arr, N, i):
        largest = i  # Initialize largest as root
        for j in range(1, self.d + 1):
            child = self.d * i + j
            if child < N and arr[largest] < arr[child]:
                largest = child

        # Change root, if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]  # swap
            # Heapify the root.
            self.MAXheapify(arr, N, largest)

    def MINheapify(self, arr, N, i):
        smallest = i  # Initialize smallest as root
        for j in range(1, self.d + 1):
            child = self.d * i + j
            if child < N and arr[smallest] > arr[child]:
                smallest = child

        # Change root, if needed
        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]  # swap
            # Heapify the root.
            self.MINheapify(arr, N, smallest)

    def MAXheapSort(self, arr):
        N = len(arr)

        # Build a max heap.
        for i in range(N // self.d - 1, -1, -1):
            self.MAXheapify(arr, N, i)

        # One by one extract elements
        for i in range(N - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]  # swap
            self.MAXheapify(arr, i, 0)

    def MINheapSort(self, arr):
        N = len(arr)

        # Build a min heap.
        for i in range(N // self.d - 1, -1, -1):
            self.MINheapify(arr, N, i)

        # One by one extract elements
        for i in range(N - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]  # swap
            self.MINheapify(arr, i, 0)
quicksort
# Example usage:
arr = []
total_time = []
start = 0
stop = 0
rang = [1,10,100,1000,10000,100000,1000000]
for el in rang:
    start = time.time()-start
    for ele in range(1,el):
        arr.append(random.randint(1,el))
    d = 4  # Set the le of the heap
    heap = Heap(d)
    heap.MAXheapSort(arr)
    stop = time.time()-stop
    total_time.append(stop - start)
plt.plot(rang, total_time, marker='o')
plt.xlabel('Dimensione dell\'input')
plt.ylabel('Tempo di esecuzione (secondi)')
plt.title('Analisi delle prestazioni')
plt.grid(True)
plt.show()


#heap.MINheapSort(arr)
#print("MINHeap sort (d-ary): " + str(arr))
