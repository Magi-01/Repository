from numba import jit
import timeit
import random

@jit
def f(n):
    res = 0
    for i in range(n):
        res += i**2 + 2*i + 1
    return res

def g(n):
    res = 0
    for i in range(n):
        res += i**2 + 2*i + 1
    return res

@jit
def f_slow(lst):
    res = ""
    for x in lst:
        res += str(x)
    return len(res)

def g_slow(lst):
    res = ""
    for x in lst:
        res += str(x)
    return len(res)


print(f"Esecuzione di g: {timeit.timeit(lambda: g(100))}")
print(f"Esecuzione di f: {timeit.timeit(lambda: f(100))}")


lst = [random.random() for _ in range(10)]

print(f"Esecuzione di g_slow: {timeit.timeit(lambda: g_slow(lst))}")
print(f"Esecuzione di f_slow: {timeit.timeit(lambda: f_slow(lst))}")
