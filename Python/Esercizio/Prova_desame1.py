class ExamException(Exception):
    pass


class ExamException(Exception):
    pass


class MovingAverage:
    def __init__(self,window):
        self.window = window

    def compute(self,data):
        if data == [] or self.window == int or self.window < 1:
            raise ExamException('Errore, lista vuota')

        if not isinstance(data,list) and not isinstance(data,tuple) and not isinstance(data,dict):
            return data

        new_data = []

        if data is not list:
            for item in data:
                try:
                    item = int(item)
                except Exception as e:
                    raise ExamException(f'Exception raised as {e}')
                new_data.append(item)
            data = new_data
        
        if self.window >= len(data):
            return data

        avg = []
        wind = self.window

        for i in range(len(data)):
            if self.window <= len(data):
                k = sum(data[i:self.window])
                avg.append(k/wind)
                self.window = self.window + 1
        if avg == []:
            return data
        return avg

mavg=MovingAverage(2)
print(mavg.compute([2, 4, 8, 16, 32]))