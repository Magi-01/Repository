class Divisibility():
    def __init__(self):
        pass

    def eval(self):
        a = []
        for i in range(0,839882):
            n = 56*i +(4*i +1)
            if n % 7 == 0:
                a.append(n)
            i = i + 1
        return a

value = Divisibility()
final = value.eval()
print([item for item in final])