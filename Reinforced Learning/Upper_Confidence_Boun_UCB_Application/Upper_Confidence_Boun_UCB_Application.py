# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 
import random 

data = pd.DataFrame(pd.read_csv("C:\\Users\\Arif Furkan\\OneDrive\\Belgeler\\Python_kullanirken\\Ads_CTR_Optimisation.csv"))
print(data)

#Random Selection
N = 10000
d = 10
total = 0
chosen = []
for n in range(0,N):
    ad = random.randrange(d)
    chosen.append(ad)
    award = data.values[n,ad]
    total = total + award
print('Toplam Odul:') 
print(total)

plt.hist(chosen)
plt.show()


#UCB
N=10000
d=10
awards = [0] * d
clicks=[0] * d
total = 0 
chosen = []
for n in range(1,N):
    ad = 0 
    max_ucb = 0
    for i in range(0,d):
        if(clicks[i] > 0):
            ortalama = awards[i] / clicks[i]
            delta = math.sqrt(3/2* math.log(n)/clicks[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: 
            max_ucb = ucb
            ad = i          
    chosen.append(ad)
    clicks[ad] = clicks[ad]+ 1
    award = data.values[n,ad] 
    awards[ad] = awards[ad]+ award
    total = total + award
print('Toplam Odul:')   
print(total)

import math
#UCB
N = 10000 
d = 10  
#Ri(n)
awards = [0] * d 
#Ni(n)
clicks = [0] * d 
total = 0 
chosen = []
for n in range(1,N):
    ad = 0 
    max_ucb = 0
    for i in range(0,d):
        if(clicks[i] > 0):
            ortalama = awards[i] / clicks[i]
            delta = math.sqrt(3/2* math.log(n)/clicks[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: 
            max_ucb = ucb
            ad = i          
    chosen.append(ad)
    clicks[ad] = clicks[ad]+ 1
    award = data.values[n,ad] 
    awards[ad] = awards[ad]+ award
    total = total + award
print('Toplam Odul:')   
print(total)

plt.hist(chosen)
plt.show()