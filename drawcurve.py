import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#get the data I need,position on x-y koordinate,and the total velocity at each time step
with open('allAmsterdamerRing.pickle','rb') as fp:
    data = pickle.load(fp)
curve_data = np.array(data[1300])
cd = np.array(curve_data[1])
pos_x = cd[:,2]
pos_y = cd[:,3]
plt.figure(figsize=(7,7))
plt.xlim((-50,50))
plt.ylim((-50,50))
plt.scatter(pos_x,pos_y,s=1)
plt.show()

