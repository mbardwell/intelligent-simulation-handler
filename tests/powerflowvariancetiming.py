# -*- coding: utf-8 -*-
"""
Plotting the effect of variance on power flow time

@author: mikey
"""

import sys # import for utils 
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import time

from utils import import_zp
from montecarlo import generateMonteCarloBinaries, generateJson
from powerflowsim import PowerFlowSim

def calculateVariance(filename, length = 1000):
    sim = import_zp(filename)
    return np.var(sim['load']['profile'][0:length])

def viewLoadProfile(filename, length = 1000):
    import matplotlib.pyplot as plt
    sim = import_zp(filename)
    plt.plot(sim['load']['profile'][0:length])
    
def plotit(x, y):
    import matplotlib.pyplot as plt
    from textwrap import wrap
    fit = np.polyfit(x,y,1)
    fit_fn = np.poly1d(fit)        
    plt.plot(x, y, 'yo', x, fit_fn(x), '--k')
    plt.legend(['Real', 'Regression'])
    
    plt.ylabel('Runtime (s)')
    plt.xlabel('Variance')
    title = 'Power Flow Simulation Runtime vs Load Variance for a 100 Node Radial Network'
    plt.title('\n'.join(wrap(title,50)))
    plt.savefig('pfs_simtime.pdf', transparent = True)
    plt.show()
    
no_houses = 100
generateJson(no_houses, 'mikework')
simlength = 100
data = [[], []]; k = 0
for i in np.arange(0.1, 10, 0.5):
    generateMonteCarloBinaries('C:/Users/mikey/Downloads/zp/18356', 0, no_houses, i, 'mikework')
    
    pfs = PowerFlowSim(simlength, 'radial', '../_configs/montecarlo' + str(no_houses) + '.json')
    before = time.time();
    pfs.nrPfSim()
    after = time.time(); 
    data[1].append(after-before); 
    
    var = []
    for j in range(no_houses):
        var.append(calculateVariance('C:/Users/mikey/Downloads/zp/montecarlo' + str(j), simlength))
    data[0].append(np.mean(var));
    print('Runtime (s): ', data[1][k], ' Variance: ', data[0][k]); k += 1
    
plotit(data[0], data[1])