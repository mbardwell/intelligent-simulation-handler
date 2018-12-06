# -*- coding: utf-8 -*-
"""
instantiate power flow sim class ----- pfs
instantiate ANN class ----- ann_regression
instantiate NE class ----- normaleqn_regression
run ann_regression and normaleqn_regression on pfs --out-> mae, mae, topology
re-run pfs with different characteristics

@author: Michael
"""

import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

import sys, os
sys.path.append('../'); sys.path.append('../../')
from regression_tools import TrainANN, NormalEquation
from powerflowsim import PowerFlowSim
from montecarlo import generateJson
    
def trainANN(nodeloads, nodevoltages):
    trainer = TrainANN(nodeloads, nodevoltages)
    trainer.buildModel()
    trainer.trainModel()
    return trainer.evaluateModel()

def normalEquation(nodeloads, nodevoltages):
    netrainer = NormalEquation(nodeloads, nodevoltages)
    netrainer.calculateTheta()
    netrainer.calculateBias()
    netrainer.buildModel()
    return netrainer.evaluateModel()

def plotIt(x, y1, y2, ylim = 0):
    fit1 = np.polyfit(x, y1, 1)
    fit2 = np.polyfit(x, y2, 1)
    fit1_fn = np.poly1d(fit1)
    fit2_fn = np.poly1d(fit2)
    plt.plot(x, y1, 'yo', x, fit1_fn(x), '--y', 
             x, y2, 'bo', x, fit2_fn(x), '--b')
    plt.legend(['Real ANN', 'Regression ANN', 'Real NE', 'Regression NE'])
    if ylim:
        plt.ylim(0, ylim)
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of Nodes')
    title = '''Neural Network Regression vs Normal Equation for 
                Expanding Radial Networks
            '''
    plt.title('\n'.join(wrap(title,50)))
    plt.savefig('pfs_regression_comparison.pdf', transparent = True)
    plt.show()
    
def checkForJson(i):
    subpath = '_configs/montecarlo' + str(i) + '.json'
    path = os.path.join('C:/Users/Michael/Documents/etx_fullscale_mike/', subpath)
    try: 
        open(path)
    except:
        generateJson(i)

data = [[], [], []]
for i in range(80, 120, 4):
    checkForJson(i)
    pfs = PowerFlowSim(500, 'radial', '../_configs/montecarlo' + str(i) + '.json')
    pfs.nrPfSim(showall = False)
    
    annmae = trainANN(pfs.nodeloads, pfs.nodevoltages.T[1:].T)
    nemae = normalEquation(pfs.nodeloads, pfs.nodevoltages)
    
    data[0].append(i); print('-------data[0]:', data[0])
    data[1].append(annmae); print('-------data[1]:', data[1])
    data[2].append(nemae); print('-------data[2]:', data[2])
print(data) 
plotIt(data[0], data[1], data[2])
