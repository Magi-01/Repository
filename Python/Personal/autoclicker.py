import matplotlib.pyplot as plt
import time, random


class MergeSort:
    def merge(self, l, r):
        k = []
        i,j = 0,0

        while i < len(l) and j < len(r):
            if l[i] < r[j]:
                k.append(l[i])
                i += 1
            else:
                k.append(r[j])
                j += 1

        while i < len(l):
            k.append(l[i])
            i += 1

        while j < len(r):
            k.append(r[j])
            j += 1
        return k

    def merge_sort_out_of_place(self, a):
        if len(a) <= 1:
            return a

        mid = int(float(len(a) / 2))
        l = a[:mid]
        r = a[mid:]

        l = self.merge_sort_out_of_place(l)
        r = self.merge_sort_out_of_place(r)

        return self.merge(l, r)
    
    def shift(self, a, l, r):
        if l < r < len(a):
            t = a[r]
            for i in range(r, l, -1):
                a[i] = a[i - 1]
            a[l] = t

    def merge_sort_in_place(self, a, start, end):
        if start >= end:
            return
        mid = int(float((start+end)/2))
        MergeSort.merge_sort_in_place(self,a,start,mid)
        MergeSort.merge_sort_in_place(self,a,mid+1,end)

        l = start
        r = mid+1

        while l <= mid and r <= end:
            if a[l] <= a[r]:
                l += 1
            else:
                MergeSort.shift(self, a, l, mid+1)
                l += 1
                mid += 1
                r += 1
        return a

inn = [1, 10, 100, 1000, 10000]
total_time = []
for element in inn:
    start = time.time()
    sort = MergeSort()
    range_element = [int(float(el)) for el in range(element)]
    range_element = random.choices(range_element, k=element)
    out = sort.merge_sort_in_place(range_element,0,int(float(len(range_element)-1)))
    stop = time.time()
    total_time.append(stop - start)
plt.plot(inn, total_time, marker='o')
plt.xlabel('Dimensione dell\'input')
plt.ylabel('Tempo di esecuzione (secondi)')
plt.title('Analisi delle prestazioni')
plt.grid(True)
plt.show()
for element in out:
    print(element,"\n")
total_time = []
for element in inn:
    start = time.time()
    sort = MergeSort()
    range_element = [int(float(el)) for el in range(element)]
    range_element = random.choices(range_element, k=element)
    out = sort.merge_sort_out_of_place(range_element)
    stop = time.time()
    total_time.append(stop - start)
plt.plot(inn, total_time, marker='o')
plt.xlabel('Dimensione dell\'input')
plt.ylabel('Tempo di esecuzione (secondi)')
plt.title('Analisi delle prestazioni')
plt.grid(True)
plt.show()
for element in out:
    print(element,"\n")