from ctypes import *

code = cdll.LoadLibrary("code.so")

class POINT(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int)]

p = POINT(10, 20)
print(f"p.x = {p.x}, p.y = {p.y}")
    
code.dist.argtypes = [POINT]
code.dist.restype = c_int
d = code.dist(p)
print(f"d = {d}")



a = 2
b = 3
x = code.sum(a, b)
print(f"{a} + {b} = {x}")

a = 2
b = 5.6
code.multiply.argtypes = [c_int, c_float]
code.multiply.restype = c_float
x = code.multiply(a, b)
print(f"{a} * {b} = {x}")

a = c_int(3)
b = c_int(7)
code.swap.argtypes = [POINTER(c_int), POINTER(c_int)]
pa = pointer(a)
pb = pointer(b)
print(f"a = {pa.contents}, b = {b}")
code.swap(pa, pb)
print(f"a = {a}, b = {b}")

#x = POINTER(c_int)()
#print(x.contents)
