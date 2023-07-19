import datetime
import matplotlib.pyplot as plt
import random

class DataConstuct:
    def __init__(self):
        self.a = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.a == datetime.datetime.now().year:
            raise StopIteration
        self.a += 1
        return random.randint(self.a, datetime.datetime.now().year)

class EpochConstruct:
    def __init__(self,data):
        self.data = data

    def epoch(self):
        i = 1
        c = []
        for item in self.data:
            if i % 4 != 0:
                item = item * 31536000
            elif i % 4 == 0:
                item = item * 31622400
            i += 1
            c.append(item)
        return c

year_start = DataConstuct()
year_1 = iter(year_start)
a = []
while True:
    try:
        a.append(next(year_1))
    except StopIteration:
        break

epoch_start = EpochConstruct(a)
epoch_1 = epoch_start.epoch()
for item in a:
    if item % 4  != 0:
        print("NB Bimestral")
    elif item % 4 == 0:
        print("Bimestral")
plt.plot(epoch_1, color='tab:red')
plt.plot(a,color='tab:blue')
plt.plot()
plt.show()