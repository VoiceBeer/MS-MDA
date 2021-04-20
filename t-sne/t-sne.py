'''
Description: 
Author: voicebeer
Date: 2021-03-15 08:03:30
LastEditTime: 2021-03-16 00:44:03
'''

%matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

sinplot()