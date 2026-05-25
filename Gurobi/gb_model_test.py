import numpy as np
import gurobipy as gp
from gurobipy import GRB

m = gp.Model("max Profit")

c_r = 5
c_n = 7
x = m.addMVar(3, lb = [0]*3, vtype=GRB.INTEGER, ub=GRB.INFINITY)
p = np.array([45, 40, 35])
c_i = np.array([[12, 6, 4], [8, 10, 3]])
c_p_i = np.array([[30, 30, 10], [40, 20, 20]])  # Adjusted structure
x_i_j = np.array([[1, 0], [0, 1]])
I1 = np.array([
    [2,0,0],
    [0,1,3],
    [0,0,2]
])
I2 = np.array([
    [4,0,0],
    [0,2,2],
    [0,0,5]
])

cstr1 = m.addConstr(sum(I1[i]@x for i in range(3)) <= 200)
cstr2 = m.addConstr(sum(I2[i]@x for i in range(3))<= 200)

cstr3 = m.addConstrs(I1[i]@x <= c_p_i[0,i] for i in range(3))
cstr3 = m.addConstrs(I2[i]@x <= c_p_i[1,i] for i in range(3))
print(I1[1]@[x[0],x[1],x[2]])
cstr4 = m.addConstr(I1[0]@x[0] + I2[0]@x[0] >= 2 * I1[1]@x[0] + 2 * I2[1]@x[0])

cstr5 = m.addConstr(I1[0]@x[0] + I2[0]@x[0] <= I1[2]@x[2] + I2[2]@x[2])


m.setObjective(sum(I1[i]@x * (p[i] - (c_i[j, i] + c_r * I1[j, i] + c_n * I1[j, i])) for j in range(3) for i in range(2))+sum(I2[i]@x * (p[i] - (c_i[j, i] + c_r * I2[j, i, 0] + c_n * I2[j, i, 1])) for j in range(3) for i in range(2)), GRB.MAXIMIZE)

m.optimize()

print('Optimal Profit:', m.objVal)
