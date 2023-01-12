from matplotlib import pyplot

name = 'shampoo_sales.csv'

class CSVFile():
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
              print(f"\n\nErrore manca valore in {line}!\n\n")
              pass
    my_file_use.close()
    if data is None:
      return None
    else:
      return data


class Model():
    def fit(self,data):
        raise NotImplementedError("Method not Implemented on Model")
    def predict(self,data):
        raise NotImplementedError("Method not Implemented on Model")


class FitIncrementModel(Model):
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
          print(f"sum: {summ}")
        except Exception as e:
          raise Exception("You stupid")
        self.gbl_avrg_inc = summ/(len(data)-1)
        return self.gbl_avrg_inc

    def predict(self,data):
        prev_data = self.gbl_avrg_inc
        prev_val = None
        summ = 0
        for item in data:
            if prev_val is None:
                prev_val = item
            summ = (item-prev_val) + summ
            prev_val = item
        current_data = summ/(len(data)-1)
        prediction = data[-1] + (current_data+prev_data)/2
        return prediction

    
       

data_got = CSVFile(name)
data = data_got.get_data()

read_data = FitIncrementModel()
read_data.fit(data)
predictions = read_data.predict(data)
print(predictions)
pyplot.plot(data+[predictions], color='tab:red')
pyplot.plot(data,color='tab:blue')
pyplot.show()