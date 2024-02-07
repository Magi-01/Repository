name = "C:/Users/mutua/Documents/Repository/Python/Esame/data.csv"

#Program that takes a file as input (given the file has two digits separated by a comma representing epoch and temp) and return the __ per every epoch


#Function that catches Exception and, in this case, passes them to
#class Exception
class ExamException(Exception): 
    pass

#Class that takes an epoch,temp file, reads it and only copies
#the lines with temp readings and epochs
class CSVTimeSeriesFile:
    #initiation
    def __init__(self,name): 
        self.name = name


    def get_data(self):
        #Try to see if the file is readable or existant
        #if not raise ExamException
        try:
            file_opn = open(self.name,'r')
            file_opn.readline()
        except:
            raise ExamException("file is either unreadable or noneexistant")
        file_opn.close() #closes file as to not have it in memroy


        #copies the lines of the file except
        #the lines that do not have numbers
        data = []
        self.file = open(self.name,'r')
        for line in self.file:
            line = line.split(',')
            line[-1] = line[-1].strip()
            if line[0] != "epoch":
                c = []
                for element in line:
                    try:
                        float(element)
                        c.append(element)
                    except:
                        pass
                data.append(c)
        self.file.close() #closes file as to not have it in memroy

        #if the previous function was not able to copy anything,
        #then raise Exception
        if data == []:
            raise ExamException ("file is empty or has errors")

        #For safety, makes sure that the copiesd files are
        #1.) digits
        #2.) the digits are integer (or convertable to integers) for epochs
        #3.) the digits are integer or float for epochs

        comp = []
        for element in data:
            try:
                int(element[0])
            except:
                pass
            try:
                float(element[1])
                comp.append(element)
            except:
                pass
        data = comp
        
        #if the epochs are or the temp are not convertable,
        #raise Exception
        if data == []:
            raise ExamException ("file is does not contain epochs and/or temperature per line")
        
        #checks if the epochs are in order
        i=0
        prev_val = 0
        for element in data:
            if i == 0:
                prev_val = element[0]
                i += 1
            else:
                if element[0] <= prev_val:
                    #raises exception if not in order
                    raise ExamException("epochs not in order")
                else:
                    prev_val = element[0]
        return data

#Function for calculating the per day
def compute_daily_max_difference(time_series):

    #if the length of the data is 1, return None as it is impossible
    #to calculate a difference
    if len(time_series) == 1:
        return None

    epoch = []
    temprt = []

    #separates and transforms the epochs (to integer if they were not)
    #and the temp
    for element in time_series:
        epoch.append(int(element[0]))
        temprt.append(float(element[1]))
    
    #function for calculating the beginning of a day
    day_start = epoch[0] - (epoch[0] % 86400)
    temp_diff = []


    #if the length of the data is 2, you only need to compare the
    #epochs and if same day, give out the difference else, 
    #you have a list of None
    if len(epoch) == 2:
        if epoch[1] - epoch[0] >= 86400:
            temp_diff = [None,None]
            return temp_diff
        else:
            temp_diff.append(temprt[1]-temprt[0])
            return temp_diff

    
    count = 0
    max = temprt[0]
    min = temprt[0]
    if len(epoch) > 2:
        for i in range(len(epoch)):
            #separated into four cases for ease
            
            #the first epoch
            if i == 0:
                #if the second element and first element of epoch
                #are on the same day, then
                #max_temp and min_temp = first element of temp
                if epoch[i+1] - epoch[i] < 86400:
                    max = temprt[i]
                    min = temprt[i]
                #else; the first element of __ is None
                #while the max_temp and min_temp = second element of temp
                else:
                    m = None
                    max = temprt[i+1]
                    min = temprt[i+1]
                    day_start = epoch[i+1] - (epoch[i+1] % 86400)
                    temp_diff.append(m)

            elif i == 1:
                #if the third element and second element of epoch
                #or the second and first are on the same day, then
                #max_temp and min_temp = max between second and first
                #element of temp
                if epoch[i+1] - epoch[i] < 86400 or epoch[i] - epoch[i-1] < 86400:
                    if max < temprt[i]:
                        max = temprt[i]
                    if min > temprt[i]:
                        min = temprt[i]
                #else finds the max between the first element and the second
                #(same for min), and if and only if the current and previous
                #epochs are on the same day, then the __ = max - min
                #else the __ is None
                else:
                    if max < temprt[i]:
                        max = temprt[i]
                    if min > temprt[i]:
                        min = temprt[i]
                    if epoch[i] - epoch[i-1] < 86400:
                        m = max - min
                    else:
                        m = None
                    max = temprt[i+1]
                    min = temprt[i+1]
                    day_start = epoch[i+1] - (epoch[i+1] % 86400)
                    temp_diff.append(m)
            
            #in betweem the second and last epoch
            elif i > 1 and i < len(epoch)-1:
                #if the current epoch is part of a certain day,
                #checks if it has the max or min value of temp.
                #if it has, then max and/or min becomes the new temp
                #adds a counter
                if epoch[i] < day_start + 86400:
                    if max < temprt[i]:
                        max = temprt[i]
                    if min > temprt[i]:
                        min = temprt[i]
                    count += 1
                #if the current epoch is not part of the previous day,
                #(ending at 1min to midnight):
                elif epoch[i] >= day_start + 86400:
                    #checks if the the counter is 1 (meaning only one epoch
                    #in the previous day), and if the diff between the
                    #current epoch and previous is less than a day, if so
                    #then the __ is none
                    if count == 1 and epoch[i] - epoch[i-1] < 86400:
                        m = None
                        temp_diff.append(m)
                    #else the __ is max - min
                    else:
                        m = max - min
                        temp_diff.append(m)
                    #updates the max and min to be of the current epoch
                    min = temprt[i]
                    max = temprt[i]
                    #updates the beginning of the day
                    day_start = epoch[i] - (epoch[i] % 86400)
                    #the counter is updated to 1 if and only if the next epoch
                    #is not in the same day
                    if epoch[i+1] >= day_start + 86400:
                        count = 1
            
            #the last epoch
            elif i == len(epoch)-1:
                #if then diff between the current epoch and the prev is
                #< 86400 then checks if the max(/min) is >(/<) than
                #previous and updates respectively
                #the __ = min - max
                if epoch[i] - epoch[i-1] < 86400:
                    if max < temprt[i]:
                        max = temprt[i]
                    if min > temprt[i]:
                        min = temprt[i]
                    m = max - min
                #else, checks if the previos epoch and the one before were
                #on the same day, if they were then the __ = min - max
                #otherwise given that the previous 2 epochs were not on the
                #same day and neither is the current one, then
                #__ = [None,None]
                else:
                    if epoch[i-1] - epoch[i-2] < 86400:
                        m = max - min
                    else:
                        m = None
                    temp_diff.append(m)
                    m = None
                temp_diff.append(m)

        return temp_diff

time_series_file = CSVTimeSeriesFile(name='data.csv')
time_series = time_series_file.get_data()
for item in time_series:
    print(item)
diff = compute_daily_max_difference(time_series)         
print(diff)