import matplotlib.pyplot as plt

name = 'shampoo_sales.csv'

class CSVFile:

  def __init__(self, name):
    self.name_op = name
    self.name_read = True
    try:
      file_opened = open(self.name_op, 'r')
      file_opened.readline()
    except Exception as e:
      self.name_read = False
      print('Errore in apertura del file: "{}"'.format(e))

  def get_data(self):
    if not self.name_read:
      print('Errore, file non aperto o illeggibile')
      return None
    else:
      data = []
      my_file_use = open(self.name_op, 'r')
      for line in my_file_use:
        elements = line.split(',')
        elements[-1] = elements[-1].strip()
        if elements[0] != 'Date':
          try:
            number = int(float(elements[1]))
            data.append(number)
          except ValueError:
            raise Exception(f"\n\n valore in {line} non int!\n\n")
    my_file_use.close()
    if data is []:
      return None
    else:
      return data

  def fit(self, data):
    pass

  def predict(self, data):
    pass


class Model():

  def fit(self, data):
    raise NotImplementedError("Method not Implemented on Model")

  def predict(self, data):
    raise NotImplementedError("Method not Implemented on Model")


class FitIncrementModel(Model):

  def __init__(self):
    self.global_avg_increment = 0

  def fit(self, data):
    summ = 0
    prev_val = None
    try:
      for item in data:
        print(item)
        if prev_val is None:
          prev_val = item
        else:
          print(prev_val)
          b = item - prev_val
          summ = summ + b
          prev_val = item
    except Exception as e:
      raise Exception(f"Error of type {e}")
    incr = summ / (len(data) - 1)
    self.global_avg_increment = incr
    return self

  def predict(self, data):
    if not isinstance(data, (list, tuple)):
        raise ValueError("Input data must be a list, tuple of integers")
    if len(data) < 2:
        raise ValueError("Input data should contain at least two elements to make a prediction")
    prev_data = self.global_avg_increment
    prev_val = None
    summ = 0
    try:
        for item in data:
            if not isinstance(item, int):
                raise ValueError(f"{item} is not an integer")
            if prev_val is None:
                prev_val = item
            summ = (item - prev_val) + summ
            prev_val = item
            current_data = summ / (len(data) - 1)
            prediction = prev_val + (current_data + prev_data) / 2
    except Exception as e:
        raise e
    if prediction <= 0 or prediction is None:
        return None
    return prediction


data_read = CSVFile(name)
data_gotten = data_read.get_data()
fitincrement = FitIncrementModel()

data = FitIncrementModel.predict(data_read,[50,52,60])
print(data)