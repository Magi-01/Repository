#!/usr/bin/env python3
import sys, random, math, os

# To plot the instance, uncomment the next five lines and the twelve last lines (it deteriorates performance)
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# import matplotlib.pyplot as plt

if len(sys.argv) < 8:
    print('Missing arguments:\n\t python generate.py n depotPos custPos demandType avgRouteSize instanceID randSeed')
    help="""

    n (number of customers)

    Depot positioning
        1 = Random				
        2 = Centered				
        3 = Cornered				
                        
    Customer positioning
        1 = Random				
        2 = Clustered				
        3 = Random-clustered		
                        
    Demand distribution	
        1 = Unitary		
        2 = Small, large var		
        3 = Small, small var		
        4 = Large, large var		
        5 = Large, small var		
        6 = Large, depending on quadrant	
        7 = Few large, many small

    Average route size
        1 = Very short
        2 = Short
        3 = Medium
        4 = Long
        5 = Very long
        6 = Ultra long
        
    Output: instance file XML<n>_<depotPos><custPos><demandType><avgRouteSize>_<instanceID>.vrp

    For more details about the generation process read:
        Uchoa et al (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. European Journal of Operational Research
        Queiroga, Eduardo, et al. (2022). 10,000 optimal CVRP solutions for testing machine learning based heuristics.
        """
    print(help) 
    exit(0)

# constants
maxCoord = 1000
decay = 40
R = 6371  # Earth radius in km for 2D Earth Cartesian mapping

# read input arguments
n = int(sys.argv[1])
rootPos = int(sys.argv[2])
custPos = int(sys.argv[3])
demandType = int(sys.argv[4])
avgRouteSize = int(sys.argv[5])
instanceID = int(sys.argv[6])
randSeed = int(sys.argv[7]) # random seed for reproducibility

random.seed(randSeed)

# Helper function to calculate Euclidean distance in 2D
def distance(x,y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

# Instance name formatting
instanceName = 'XML'+str(n)+'_'+str(rootPos)+str(custPos)+str(demandType)+str(avgRouteSize)+'_'+ format(instanceID, '02d')

#Depot positioning
if rootPos == 1:
    x_ = random.randint(0,maxCoord)
    y_ = random.randint(0,maxCoord)
elif rootPos == 2:
    x_ = y_ = int(maxCoord/2.0)
elif rootPos == 3:
    x_ = y_ = 0
else:
    print("Depot Positioning out of range!")
    exit(0)
depot = (x_,y_)

#Customer positioning
if custPos == 3:
    nRandCust = int(n/2.0)
elif custPos == 2:
    nRandCust = 0
elif custPos == 1:
    nRandCust = n
else:
    print("Costumer Positioning out of range!")
    exit(0)
nClustCust = n - nRandCust

#Generating random customers
S = set()
for i in range(1, nRandCust+1):
    x_ = random.randint(0,maxCoord)
    y_ = random.randint(0,maxCoord)
    while (x_,y_) in S or (x_,y_) == depot:
        x_ = random.randint(0,maxCoord)
        y_ = random.randint(0,maxCoord)
    S.add((x_,y_))

#Generation of clustered customers
nSeeds = random.randint(2,6) 
seeds = []

if nClustCust > 0:
    if nClustCust < nSeeds:
        print("Too many seeds!")
        exit(0)
    
    #Generate the seeds
    for i in range(nSeeds):
        x_ = random.randint(0,maxCoord)
        y_ = random.randint(0,maxCoord)
        while (x_,y_) in S or (x_,y_) == depot:
            x_ = random.randint(0,maxCoord)
            y_ = random.randint(0,maxCoord)
        S.add((x_,y_))
        seeds.append((x_,y_))
    nS = nRandCust + nSeeds
    
    # Determine the seed with maximum sum of weights (w.r.t. all seeds)
    maxWeight = 0.0
    for i,j in seeds:
        w_ij = 0.0
        for i_,j_ in seeds:
            w_ij += 2**(-distance((i,j), (i_,j_)) / decay)
        if w_ij > maxWeight:
            maxWeight = w_ij

    norm_factor = 1.0/maxWeight

    # Generate the remaining customers using Accept-reject method
    while nS < n:
        x_ = random.randint(0,maxCoord)
        y_ = random.randint(0,maxCoord)
        while (x_,y_) in S or (x_,y_) == depot:
            x_ = random.randint(0,maxCoord)
            y_ = random.randint(0,maxCoord)
        
        weight = 0.0
        for i_,j_ in seeds:
            weight += 2**(-distance((x_,y_), (i_,j_)) / decay)
        weight *= norm_factor
        rand = random.uniform(0,1)

        if rand <= weight: # Will we accept the customer?
            S.add((x_,y_))
            nS += 1

V = [depot] + list(S) # set of vertices (from now on, the ids are defined)

# Demands
demandMinValues = [1,1,5,1,50,1,51,50,1]
demandMaxValues = [1,10,10,100,100,50,100,100,10]
demandMin = demandMinValues[demandType-1]
demandMax = demandMaxValues[demandType-1]

D = [] # demands
sumDemands = 0
maxDemand = 0

In = {1:(3,5), 2:(5,8), 3:(8,12), 4:(12,16), 5:(16,25), 6:(25,50)}
r = random.uniform(In[avgRouteSize][0], In[avgRouteSize][1])

for i in range(2,n + 2):
    j = int((demandMax - demandMin + 1) * random.uniform(0,1) + demandMin)
    if demandType == 6:
        if (V[i - 1][0] < maxCoord/2.0 and V[i - 1][1] < maxCoord/2.0) or (V[i - 1][0] >= maxCoord/2.0 and V[i - 1][1] >= maxCoord/2.0):
            j = int((100 - 51 + 1) * random.uniform(0,1) + 51)
    D.append(j)
    sumDemands += j
    if j > maxDemand:
        maxDemand = j

# Generate capacity
capacity = max(maxDemand, math.ceil(r * sumDemands / n))
k = math.ceil(sumDemands/float(capacity))

# Map 2D CVRP coordinates to Earth 2D Cartesian
def to_earth_cartesian(x, y):
    lat = (y / maxCoord) * 180 - 90
    lon = (x / maxCoord) * 360 - 180
    x_cart = R * math.radians(lon) * math.cos(math.radians(lat))
    y_cart = R * math.radians(lat)
    return x_cart, y_cart

# Write VRP file
with open(instanceName+'.vrp','w') as f:
    f.write('NAME : ' + instanceName + '\n')
    f.write('COMMENT : Generated as the XML100 dataset from the CVRPLIB, Earth-scaled 2D\n')
    f.write('TYPE : CVRP\n')
    f.write('DIMENSION : ' + str(n+1) + '\n')
    f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
    f.write('CAPACITY : ' + str(int(capacity)) + '\n')
    f.write('NODE_COORD_SECTION\n')
    for i,v in enumerate(V):
        x_cart, y_cart = to_earth_cartesian(v[0], v[1])
        f.write('{:<4} {:<10.2f} {:<10.2f}\n'.format(i+1, x_cart, y_cart))
    f.write('DEMAND_SECTION\n')
    D = [0] + D
    for i,d in enumerate(D):
        f.write('{:<4} {:<4}\n'.format(i+1,d))
    f.write('DEPOT_SECTION\n1\n-1\nEOF\n')

# Plot
from matplotlib import pyplot as plt
x_plot, y_plot = zip(*(to_earth_cartesian(v[0],v[1]) for v in V))
x_s_plot, y_s_plot = zip(*(to_earth_cartesian(v[0],v[1]) for v in seeds))
plt.figure(figsize=(20, 20), dpi=80)
plt.scatter(x_plot, y_plot, marker='o', color='blue',edgecolor='blue', s=40)
plt.scatter(x_s_plot, y_s_plot, marker='o', color='magenta',edgecolor='magenta', s=40)
plt.scatter([x_plot[0]], [y_plot[0]], marker='s', edgecolor='black', color='yellow', s=200)
plt.xticks([]) # Turn off x labels
plt.yticks([]) # Turn off y labels
plt.savefig(instanceName+'.png')
plt.close()
