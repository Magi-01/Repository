class ExamException(Exception):
    pass


class MovingAverage:
    def __init__(self,window):
        if not isinstance(window,int):
            raise ExamException('Errore, lista vuota')
        if window < 1:
            raise ExamException('Errore, lista vuota')
        self.window = window

    def compute(self,data):
        if data == [] or data == None or not isinstance(self.window,int):
            raise ExamException('Errore, lista vuota')

        if not isinstance(data,list):
            if self.window > 1:
                raise ExamException('Errore, lista vuota')
            return data

        if self.window > len(data):
            raise ExamException('Errore, lista vuota')

        if self.window == 0:
            raise ExamException('Errore, lista vuota')

        if self.window == 1:
            return data
            
        if self.window < 0:
            self.window *= -1

        avg = []
        wind = self.window

        for i in range(len(data)):
            if wind <= len(data):
                k = sum(data[i:wind])
                avg.append(k/self.window)
                wind = wind + 1
        if avg == []:
            return data
        return avg

mavg = MovingAverage(5)
#self.assertEqual(mavg.compute([2,4,8,16]), [3,6,12])
print(mavg.compute([2,4,8,16,32]))