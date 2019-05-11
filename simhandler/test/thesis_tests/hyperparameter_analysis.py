# -*- coding: utf-8 -*-
"""
Talos-based hyperparameter search

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import sys
import datetime
import json
import numpy as np

import matplotlib.pyplot as plt
import talos as ta
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam, Nadam
from keras.activations import sigmoid
from keras.losses import mse
from keras import backend

sys.path.append('../../../')
from simhandler.powerflowsim import PowerFlowSim
from simhandler.generic import sine_fun_for_talos


"""
Test function
"""


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def build_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation=params['activation']))

    model.add(Dropout(params['dropout']))
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],
                                                params['optimizer'])
                                                ),
                  loss=params['loss'],
                  metrics=[rmse])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=early_stopper(
                            params['epochs'],
                            monitor='val_loss',
                            min_delta = 0.001,
                            )
                    )

    return out, model


p = {'lr': [1],
      'first_neuron': [1, 2, 3],
      'batch_size': [200],
      'epochs': [500],
      'dropout': [0],
      'weight_regulizer': [None],
      'optimizer': [Adam],
      'loss': ['mse'],
      'activation': [sigmoid]}  # relu

 # pfs = PowerFlowSim(100, '../../data/network_configurations/3node.json')
 # x = pfs.node_loads
 # y = pfs.node_voltages

x, y = sine_fun_for_talos(True)

h = ta.Scan(x, y,
           params=p,
           dataset_name='first_test',
           experiment_no='1',
           model=build_model,
           grid_downsample=1)
