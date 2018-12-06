# -*- coding: utf-8 -*-
"""
Linear regression tools for Power Flow emulation

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import sys
import datetime
import json
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
import numpy as np
import h5py

class TrainANN(object):
    """Trains feedforward ANN using power system load flow load/voltage data"""

    def __init__(self, load_profile=None, voltage_profile=None,
                 train_percentage=0.7, learning_rate=0.001, no_hidden_layers=1,
                 layer_density=64, dropout=False, no_epochs=1000,
                 early_stop=True, save_model=False,
                 plot_results=False):

        if load_profile is not None and voltage_profile is not None:
            _split_index = int(train_percentage * len(load_profile))
            self.train_data = load_profile[0:_split_index]
            self.train_labels = voltage_profile[0:_split_index]

            self.test_data = load_profile[_split_index+1:]
            self.test_labels = voltage_profile[_split_index+1:]

            self.buildModel(learning_rate, no_hidden_layers, layer_density,
                            dropout)
            self.trainModel(no_epochs, early_stop)
            self.evaluateModel()
            self.predictWithModel()
            if save_model:
                self.model_name = 'ann_model_' + str(datetime.datetime.now()).\
                               replace(':', '-').replace(' ', '_')
                self.saveModel(self.model_name)

    def buildModel(self, learning_rate=0.001, no_hidden_layers=1,
                   layer_density=64, dropout=False):
        """
        :rtype self.model: class 'tensorflow.python.keras.engine.sequential.Sequential'
        """
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(
            layer_density,
            activation=tf.nn.relu,
            input_shape=(self.train_data.shape[1],)))
        for _ in range(1, no_hidden_layers):
            print('here')
            if dropout:
                try:
                    self.model.add(keras.layers.Dropout(dropout))
                except BaseException as ex:
                    print('Drop not added to network. Must be a number\
                          between 0-1. {}'.format(ex))
            self.model.add(keras.layers.Dense(layer_density,
                                              activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(self.train_labels.shape[1]))

        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['mae'])
#        self.model.summary() # for debug

    def trainModel(self, no_epochs=1000, early_stop=False, _patience=200):
        """Trains ANN using Tensorflow backend"""

        if early_stop is not False:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=_patience)
            self.history = self.model.fit(self.train_data, self.train_labels,
                                          epochs=no_epochs,
                                          validation_split=0.2,
                                          verbose=0, callbacks=[early_stop])
        else:
            self.history = self.model.fit(self.train_data, self.train_labels,
                                          epochs=no_epochs,
                                          validation_split=0.2, verbose=0)

    def evaluateModel(self):
        """Evalutes keras ann model against test data"""

        [loss, mae] = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
        print("ANN regression loss: {}, mae: {}".format(loss, mae))
        return mae

    def predictWithModel(self, plot_results=True):
        """Makes predictions by applying learned ANN model on test data"""

        test_predictions = self.model.predict(self.test_data)

        if plot_results:
            plt.plot(self.test_labels, test_predictions, 'o')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.axis('equal')
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.plot([-100, 100], [-100, 100])
            plt.show()

            # Histogram
            error = test_predictions - self.test_labels
            for i in range(len(error[0])):
                plt.hist(error.T[i], bins=50)
            plt.xlabel("Prediction Error")
            plt.ylabel("Count")
            plt.show()

    def plotHistory(self):
        """Plot learning curve"""
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(self.history.epoch, np.array(self.history.history['mean_absolute_error']),
                 label='Train Loss')
        plt.plot(self.history.epoch, np.array(self.history.history['val_mean_absolute_error']),
                 label='Val loss')
        plt.legend()
        plt.ylim([0, 0.2])
        plt.show()

    def saveModel(self, name='annmodel'):
        """Save learned ANN model"""
        
        ## TO DO: Test this function
        model_json = self.model.to_json()
        with open('./data/lookup_tables/' + name + ".json", "w") as file:
            json.dump(model_json, file)
        file.close()

        # serialize weights to HDF5
        self.model.save_weights('./data/lookup_tables/' + name + ".h5")

    def loadModel(self, model_name):
        """Decodes a JSON file into a keras model"""

        path = './data/lookup_tables/'
        try:
            with open(path + model_name + '.json', 'r') as ann_model_json:
                model_json_string = ann_model_json.read().\
                replace('\\', '')[1:-1]
                model = model_from_json(model_json_string)
            ann_model_json.close()
            model.load_weights(path + model_name + '.h5', by_name=False)
            print('Opening ANN-derived look up table')
            return model
        except BaseException as ex:
            print('Line {} - lookup table loading failed. {}'.format(
                sys.exc_info()[2].tb_lineno, ex))
            return False


class NormalEquation(object):
    """Trains parametric algorithm using power system load flow load/voltage
       data
    """

    def __init__(self, load_profile=None, voltage_profile=None,
                 save_model=False):

        if load_profile is not None and voltage_profile is not None:
            self.lpf = {'data': load_profile, 'target': voltage_profile}
            m = self.lpf['target'].shape[0]
            self.lpf['data_pb'] = np.c_[np.ones((m, 1)), self.lpf['data']]
            self.calculateTheta()
            self.calculateBias()
            self.predictWithModel(load_profile)
            self.evaluateModel()

            if save_model:
                self.model_name = 'ne_model_' + \
                str(datetime.datetime.now()).\
                replace(':', '-').replace(' ', '_')
                self.saveModel(self.model_name)

    def calculateTheta(self):
        x = tf.constant(self.lpf['data_pb'],
                        dtype=tf.float32, name="x")
        y = tf.constant(self.lpf['target'], 
                        dtype=tf.float32, name="y")
        xt = tf.transpose(x)
        theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt, x)), xt),
                          y)
        with tf.Session(): ## do I even need this??
            self.theta = theta.eval()

    def calculateBias(self):
        self.bias = [self.theta[0] for i in range(self.lpf['target'].shape[0])]

    def predictWithModel(self, load_profile):
        return np.dot(load_profile, self.theta[1:].T) + self.bias

    def evaluateModel(self, plot_results=True):
        #TODO: plot results
        prediction = self.predictWithModel(self.lpf['data'])
        mse = np.mean((prediction.T[1:].T - self.lpf['target'].T[1:].T)**2)
        mae = abs(np.mean((prediction.T[1:].T - self.lpf['target'].T[1:].T)))
        print('normal equation mse: {}, mae: {}'.format(mse, mae))
        return mae

    def saveModel(self, model_name):
        """Encodes parametric model parameters into HDF5 binary data format"""

        path = './data/lookup_tables/'
        with h5py.File(path + model_name + '.h5', 'w') as file:
            file.create_dataset(name='data', data=np.array(self.theta))
        file.close()

    def loadModel(self, model_name):
        """Decodes HDF5 binary data format into parametric model parameters"""

        path = './data/lookup_tables/'
        try:
            self.theta = h5py.File(path + model_name + '.h5', 'r')['data']
            print('Opening NE-derived look up table')
            return self.theta
        except BaseException as ex:
            print('Line {} - lookup table loading failed. {}'.format(
                sys.exc_info()[2].tb_lineno, ex))
            return False


class TrainRNN(object):
    """TODO: Proof of concept only right now"""

    def __init__(self, load_profile, voltage_profile, train_percentage=0.7, 
                 name='ann_model'):
        """
        :type load_profile: List[int], voltage_profile: List[int]
        :type train_percentage: int, name: String
        """
        _split_index = int(train_percentage * len(load_profile))
        self.train_data = load_profile[0:_split_index]
        self.train_labels = voltage_profile[0:_split_index]
        print(self.train_data.shape, self.train_labels.shape)
        
        self.test_data = load_profile[_split_index+1:]
        self.test_labels = voltage_profile[_split_index+1:]
        
        self.name = name
        
    def reshape(self, data):
        """Reshape data numpy array"""
        
        return data.reshape(data.shape[0], 1, data.shape[1])


    def buildModel(self):
        """
        :rtype self.model: class 'tensorflow.python.keras.engine.sequential.Sequential'
        """
        inputdim = self.train_data.shape[1]
        hiddendim = 64
        outputdim = self.train_labels.shape[1]
        
        self.model = Sequential()
        self.model.add(Dense(units=outputdim, input_dim=inputdim))
        self.model.add(Activation("relu"))
        self.model.add(Reshape((1,inputdim)))
        self.model.add(SimpleRNN(hiddendim))
        self.model.add(Dense(units=outputdim))
        self.model.add(Activation("softmax"))
        
        self.model.compile(optimizer='rmsprop', loss = 'mse', metrics=['mae'])
    
    def trainModel(self, no_epochs = 1000):
        """
        :type model: class 'tensorflow.python.keras.engine.sequential.Sequential'
        :type epochs: int
        :rtype history: ??
        """
            
        # Store training stats
        self.history = self.model.fit(self.train_data, self.train_labels, 
                                      epochs=no_epochs,
                                      validation_split=0.2, verbose=0)
        
    def evaluateModel(self):
        [loss, mae] = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
        print("ANN regression loss: {}, mae: {}".format(loss, mae))
        return mae