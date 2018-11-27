# Utilities designed by Mike Bardwell, MSc, University of Alberta
# tied to deriving proper plots for MSc thesis 

import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

import sys; sys.path.append('../'); sys.path.append('../../'); 
import os; sys.path.append('../utils'); 
from pathlib import Path
from powerflowsim import PowerFlowSim
from montecarlo import generateJson
from regression_tools import NormalEquation
        
def plotit(error, no_bins):
    """
    Histogram plotting
    :type error: Array of mean absolute error values
    """
    
    for i in range(len(error)):
        plt.hist(error[i], bins = no_bins)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    title = 'Prediction MAE Histogram For Normal Equation \
             Regression of PSLF'
    plt.title('\n'.join(wrap(title,50)))
    plt.show()

remove_path = 'C:/Users/mikey/Downloads/zp/montecarlo' # work computer only
mae_error = [[], []]
max_no_houses = 60

for no_houses in range(5,1000,200):
    generateJson(no_houses, 'mikework', True) # binaries will be auto-gen'd
    for i in range(1): # run it 10 times
        pfs = PowerFlowSim(500, 'radial', '../_configs/montecarlo'
                           + str(no_houses) + '.json')
        pfs.nrPfSim()
        
        ne = NormalEquation(pfs.nodeloads, pfs.nodevoltages)
        ne.calculateTheta()
        ne.calculateBias()
        mae_error[0].append(no_houses)
        mae_error[1].append(ne.evaluateModel())