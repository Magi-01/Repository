"""import matplotlib.pyplot as plt
import numpy as np

class LINEARSIMPLE():
    def __init__(self, A, x, K) -> None:
        self.A = A
        self.x = x
        self.K = K
        self.obs = []

    def linear(self):
        growth = self.A @ (1 - self.x/self.K)
        self.x = self.x + self.x * growth
        return self.x

    def __iter__(self):
        return self

    def __next__(self):
        n = self.x.copy()
        self.x = self.linear()
        self.obs.append(n)
        return n
    
a = np.array([[1,0],[0,0]])
x_y = np.array([1, 0])
k = np.full_like(x_y,100)
lin = LINEARSIMPLE(a,x_y,k)
it = iter(lin)
for _ in range(30+1):
    next(it)
print(lin.obs)

obs = np.array(lin.obs)

for i in range(obs.shape[1]):
    plt.plot(obs[:, i])

plt.xlabel("Iteration")
plt.ylabel("x")
plt.show()
mask = (obs[:,0] > 0) & (obs[:,1] > 0)
plt.plot(np.log10(obs[mask, 0]), np.log10(obs[mask, 1]), marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt

# Starting animals
x_0 = 1 # Prey
y_0 = 1 # Predator

r = 1 # Prey growth rate
K = 5 # Carrying capacity
b = 1 # Predation rate
c = 1 # Predation conversion efficiency
d = 1 # Predator death rate

obs = []

for _ in range(30):
    obs.append([x_0,y_0])
    current_x = x_0
    growth_x = r * x_0 * (1 - x_0/K)
    death_x = b * (1 - 1/(b * y_0 + 1)) * x_0

    current_y = y_0
    growth_y = c * y_0 * x_0
    death_y = d * y_0

    x_0 = current_x + growth_x - death_x
    y_0 = current_y + growth_y - death_y
obs = np.array(obs)


plt.plot(obs[:, 0], label="x")
plt.plot(obs[:, 1], label="y")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.show()

# Phase plane
plt.plot(obs[:, 0], np.log10(obs[:, 1]), marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt

# Starting animals
x_0 = 10 # Prey
y_0 = 2 # Predator

r = 0.5 # Prey growth rate
K = 5 # Carrying capacity
b = 0.1 # Predation rate
c = 0.005 # Predation conversion efficiency
d = 0.2 # Predator death rate

dt = 1e-2

obs = []
# LOTKA VOLTERA
for _ in range(30):
    obs.append([x_0,y_0])

    dx = r*x_0 - b*x_0*y_0
    dy = c*x_0*y_0 - d*y_0

    x_0 += dt*dx
    y_0 += dt*dy
obs = np.array(obs)


plt.plot(obs[:, 0], label="x")
plt.plot(obs[:, 1], label="y")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.show()

# Phase plane
plt.plot(obs[:, 0], np.log10(obs[:, 1]), marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt

# Starting animals
x_0 = 10 # Prey
y_0 = 10 # Predator

r1 = 0.5 # Prey growth rate
r2 = 0.5 # Prey growth rate
K1 = 100 # Carrying capacity
K2 = 50
b = 0.01 # Predation rate
c = 0.005 # Predation conversion efficiency
d = 0.2 # Predator death rate

obs = []
# LOGISTIC LOTKA VOLTERA
for _ in range(30):
    obs.append([x_0,y_0])
    x_next = x_0 + r1*x_0*(1 - (x_0 + b*y_0)/K1)
    y_next = y_0 + r2*y_0*(1 - (y_0 + c*x_0)/K2)

    x_0 = x_next
    y_0 = y_next
obs = np.array(obs)


plt.plot(obs[:, 0], label="x")
plt.plot(obs[:, 1], label="y")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.show()

# Phase plane
plt.plot(obs[:, 0], np.log10(obs[:, 1]), marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()"""

"""import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.4, 0.39, 0.21])
eps = 1e-6
gamma = 1.0

obs = []

def intensity(p):
    return (1 / (p + eps)) ** gamma

for _ in range(100):
    obs.append(p.copy())

    I = intensity(p)
    x, y, z = p

    def flow(a, b, Ia, Ib):
        return a * max(b - a, 0) * Ib

    xy = flow(x, y, I[0], I[1])
    yx = flow(y, x, I[1], I[0])

    xz = flow(x, z, I[0], I[2])
    zx = flow(z, x, I[2], I[0])

    yz = flow(y, z, I[1], I[2])
    zy = flow(z, y, I[2], I[1])

    dx = (yx + zx) - (xy + xz)
    dy = (xy + zy) - (yx + yz)
    dz = (xz + yz) - (zx + zy)

    p = p + np.array([dx, dy, dz])

    # normalize
    p = np.clip(p, 1e-6, None)
    p = p / p.sum()

obs = np.array(obs)

plt.plot(obs[:,0], label="A")
plt.plot(obs[:,1], label="B")
plt.plot(obs[:,2], label="C")
plt.legend()
plt.show()"""

"""# WORD SEARCH
from collections import defaultdict
def func(arr, k):
    rows = len(arr)
    cols = len(arr[0])

    def dfs(r, c, idx):
        if idx == len(k):
            return True

        if (
            r < 0 or r >= rows or
            c < 0 or c >= cols or
            arr[r][c] != k[idx]
        ):
            return False

        temp = arr[r][c]
        arr[r][c] = "#"

        found = (
            dfs(r + 1, c, idx + 1) or
            dfs(r - 1, c, idx + 1) or
            dfs(r, c + 1, idx + 1) or
            dfs(r, c - 1, idx + 1)
        )

        arr[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True

    return False

inp = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCCED"
print(func(inp, word))"""

"""import timeit

def func(arr):
    rows = len(arr)

    def dfs(i, low, high):
        if i >= len(arr) or arr[i] is None:
            return True

        if not (low < arr[i] < high):
            return False

        return (
            dfs(2*i + 1, low, arr[i]) and
            dfs(2*i + 2, arr[i], high)
        )
    for row in range(rows):
        if not dfs(row, arr[row+1], arr[row+2]):
            return False

    return True

inp =   [5,1,4,None,None,3,6]
execution_time = timeit.timeit(
    lambda: func(inp),
    number=1000
)

print(execution_time)"""

class TreeNode():
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

"""# Clone Graph
from collections import deque
def func(root):
    arr = []
    if root != None:
        visit = deque()
        visit.append(root)
            
        while visit:
            tmp = []
            for _ in range(len(visit)):
                node = visit.popleft()

                tmp.append(node.val)

                if node.left:
                    visit.append(node.left)

                if node.right:
                    visit.append(node.right)

            arr.append(tmp)
        
    return arr

root = [3,9,20,None,None,15,7]
nodes = [
    None if x is None else TreeNode(x)
    for x in root
]
for i in range(len(root)):
    if nodes[i] is None:
        continue

    left = 2*i + 1
    right = 2*i + 2

    if left < len(root):
        nodes[i].left = nodes[left]

    if right < len(root):
        nodes[i].right = nodes[right]
tree = nodes[0]
print(func(tree))"""

"""from collections import deque, defaultdict
def func(C, T):
    queue = deque([(T, 0)])  # (remaining amount, steps)
    visited = set()

    while queue:
        rem, steps = queue.popleft()

        if rem == 0:
            return steps

        if rem < 0 or rem in visited:
            continue

        visited.add(rem)

        for coin in C:
            queue.append((rem - coin, steps + 1))

    return -1

coins = [1,2,5]
T = 11

print(func(coins,T))"""
def quick(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]

    left = []
    right = []

    for interval in arr:
        if interval[1] < pivot[0]:
            left.append(interval)
        else:
            right.append(interval)

    return quick(left) + [pivot] + quick(right)