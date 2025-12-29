import time
import numpy as np

n = 1024
m = 1024
p = 1024

a = np.random.rand(n, p)
b = np.random.rand(p, m)

start = time.time()
c = np.dot(a, b)
end = time.time()

print(f"Time required {end-start}s")
