import matplotlib.pyplot as plt

name = 'shampoo_sales.csv'


class CSVFile:
  def __init__(self,name):
    self.name_op = name
    self.name_read = True
    try:
      file_opened = open(self.name_op,'r')
      file_opened.readline()
    except Exception as e:
      self.name_read = False
      print ('Errore in apertura del file: "{}"'.format(e))
    
  def get_data(self):
    if not self.name_read:
      print('Errore, file non aperto o illeggibile')
      return None
    else:
      data = []
      my_file_use = open(self.name_op,'r')
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
      
  def fit(self):
    raise NotImplementedError("Method not Implemented on Model")
    
  def predict(self):
    raise NotImplementedError("Method not Implemented on Model")

  def evaluate(self):
    raise NotImplementedError("Method not Implemented on Model")


class IncrementModel(CSVFile):
  def __init__(self,data):
    self.data24 = data[0:24]
    self.data = data[24:-1]

  def evaluate(self):
    print(self.data)
    prediction = []
    j = 0
    index = 0
    while j < len(self.data)-1:
      prediction.append(FitIncrementModel().Predict(self.data[j:3+j]))
      print(prediction)
      j = j + 1
      index += 1
      print(prediction)
    error = sum(prediction)/len(self.data)
    return prediction

  def fit(self):
    pass

  def predict (self):
    pass



class FitIncrementModel(IncrementModel):
  def __init__(self):
    self.global_avg_increment = 0

  def fit(self,data):
    summ = 0
    prev_val = None
    try:
      for item in data:
        if prev_val is None:
          prev_val = item
        else:
          b = item - prev_val
          summ = summ + b
          prev_val = item
    except Exception as e:
      raise Exception(f"Errore di tipo {e}")
    incr = summ/(len(data)-1)
    self.global_avg_increment = incr
    return self

  def Predict(self,data):
    FitIncrementModel.fit(self,data)
    prev_data = self.global_avg_increment
    prev_val = None
    summ = 0
    try:
      if len(data) > 1:
        for item in data:
          if type(item) is str:
            try:
              item = int(item)
            except Exception:
              raise Exception()
          if prev_val is None:
            prev_val = item
          summ = (item - prev_val) + summ
        prev_val = item
        current_data = summ/(len(data)-1)
        prediction = prev_val + (current_data+prev_data)/2
      elif len(self.data) == 1:
        for item in self.data:
          if type(item) is str:
            try:
              item = int(item)
            except Exception:
              raise Exception()
          keep = item
          summ = item + summ
          current_data = summ/(len(data))
          prediction = keep + (current_data+prev_data)/2
      else:
          raise Exception("Exception")
    except TypeError:
      raise Exception()
    return prediction

data_read = CSVFile(name)
data_gotten = data_read.get_data()
increment = IncrementModel(data_gotten)
fitincrement = FitIncrementModel()

data = increment.evaluate()
print(data)

for item in data_gotten[24:-1]:
  plt.plot(data+[item], color='tab:red')
  plt.plot(data_gotten[24:-1],color='tab:blue')
  plt.show()
