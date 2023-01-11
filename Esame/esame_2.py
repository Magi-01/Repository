class ExamException(Exception):
    pass

class Diff:
    def __init__(self,ratio=1):
        if ratio == None:
            raise ExamException('Errore, ratio of None type')
        if type(ratio) == str:
            raise ExamException('Errore, ratio of str type')
        if ratio <= 0:
            raise ExamException('Errore, division by 0 or <0')
        
        self.ratio = ratio


    def compute(self,data):
        if data == [] or data == None: #"""or not isinstance(self.window,int)"""
            raise ExamException('Errore, lista vuota')

        if self.ratio == 0:
            raise ExamException('Errore, divisione per 0')

        if not isinstance(data,list):
            raise ExamException('Errore, data is not list')

        if len(data) <= 1:
            raise ExamException('Errore, data is not list')

        for item in data:
            try:
                int(item) or float(item)
            except:
                raise ExamException('Errore, lista non interamente di interi')

        avg = []
        wind = 1
        k = 0

        for i in range(len(data)):
            if wind < len(data):
                for j in range (i,wind+1):
                    if k == 0:
                        k = data[j]
                    else:
                        k = k - data[j]
                avg.append((k*(-1))/(self.ratio))
                wind = wind + 1
                k = 0
        if avg == []:
            return data
        return avg
