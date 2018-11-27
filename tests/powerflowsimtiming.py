# Utilities designed by Mike Bardwell, MSc, University of Alberta
# tied to deriving proper plots for MSc thesis 

import time
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

from powerflowsim import PowerFlowSim

def powerflowsimtiming():
    data = [[], []]
    for i in range(5,1000,50):
        before = time.time(); print('-------before:', before)
        PowerFlowSim(i).nrPfSim()
        after = time.time(); print('-------after:', after)
        data[0].append(i); print('-------data[0]:', data[0])
        data[1].append(after-before); print('-------data[1]:', data[1])
        
    def plotit(x, y):
        fit = np.polyfit(x,y,1)
        fit_fn = np.poly1d(fit)        
        plt.plot(x, y, 'yo', x, fit_fn(x), '--k')
        plt.legend(['Real', 'Regression'])
        
        plt.ylabel('Runtime (s)')
        plt.xlabel('Sample Size (No. Timestamps)')
        title = 'Power Flow Simulation Runtime vs Sample Size For a 3 Node Network Connected in a Ring Topology'
        plt.title('\n'.join(wrap(title,50)))
        plt.savefig('pfs_simtime.pdf', transparent = True)
        plt.show()
        
    plotit(data[0], data[1])
    
powerflowsimtiming()