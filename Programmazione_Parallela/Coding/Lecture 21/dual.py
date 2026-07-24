class Dual:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __add__(self, other):
        if isinstance(other, Dual):
            ra = self.a + other.a
            rb = self.b + other.b
        else:
            ra = self.a + other
            rb = self.b
        return Dual(ra, rb)

    def __sub__(self, other):
        if isinstance(other, Dual):
            ra = self.a - other.a
            rb = self.b - other.b
        else:
            ra = self.a - other
            rb = self.b
        return Dual(ra, rb)

    def __mul__(self, other):
        if isinstance(other, Dual):
            ra = self.a * other.a
            rb = self.a * other.b + self.b * other.a
        else:
            ra = self.a * other
            rb = self.b * other
        return Dual(ra, rb)
        
    def __str__(self):
        return f"{self.a} + {self.b}ε"


def f(x):
    if (x.a > 3):
        return x * 2
    else:
        return x*x
    
