# -*- coding: utf-8 -*-
"""
Regression tools for power system load flow function mapping

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import sys
import os
import datetime
import json
from time import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import SimpleRNN
from keras import backend
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path

class ANNRegression():
    """Trains feedforward ANN using power system load flow load/voltage data"""

    def __init__(self, load_profile=None, voltage_profile=None,
                 train_percentage=0.7, optimizer='Adam', lr=0.001,
                 no_hidden_layers=1, layer_density=64, dropout=False,
                 batch_size=32, no_epochs=1000, early_stop=False,
                 save_model=False, plot_results=False):

        if load_profile is not None and voltage_profile is not None:
            _split_index = int(train_percentage * len(load_profile))
            self.train_data = load_profile[0:_split_index]
            self.train_labels = voltage_profile[0:_split_index]

            self.test_data = load_profile[_split_index+1:]
            self.test_labels = voltage_profile[_split_index+1:]

            self.buildModel(optimizer, lr, no_hidden_layers, layer_density,
                            dropout)
            self.trainModel(batch_size, no_epochs, early_stop)
            self.evaluateModel()
            self.predictWithModel(self.test_data, plot_results)
            if save_model:
                self.model_name = 'ann_model_' + str(datetime.datetime.now()).\
                               replace(':', '-').replace(' ', '_')
                self.saveModel(self.model_name)

    def buildModel(self, optimizer='Adam', lr=0.001, no_hidden_layers=1,
                   layer_density=64, dropout=False):
        """
        :type dropout: False (bool) or number between 0-1
        :rtype self.model: class 'keras.engine.sequential.Sequential'
        """
        
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(
            layer_density,
            activation=tf.nn.relu,
            input_shape=(self.train_data.shape[1],)))
        for _ in range(1, no_hidden_layers):
            if dropout:
                try:
                    self.model.add(keras.layers.Dropout(dropout))
                except BaseException as ex:
                    print('Drop not added to network. Must be a number\
                          between 0-1. {}'.format(ex))
            self.model.add(keras.layers.Dense(layer_density,
                                              activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(self.train_labels.shape[1]))

        if optimizer == 'RMSprop':
            opt = tf.train.RMSPropOptimizer(lr)
        elif optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(lr)
 
        def rmse(y_true, y_pred):
            return backend.sqrt(backend.mean(backend.square(y_pred - y_true), 
                                             axis=-1))        
        
        # mse used instead of rmse because it one less step
        self.model.compile(loss='mse',
                           optimizer=opt,
                           metrics=[rmse])
#        self.model.summary() # for debug

    def trainModel(self, batch_size, no_epochs=1000, early_stop=False, 
                   _patience=200):
        """Trains ANN using Tensorflow backend"""

        if early_stop is not False:
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=_patience)
            self.history = self.model.fit(self.train_data, 
                                          self.train_labels,
                                          batch_size=batch_size,
                                          epochs=no_epochs,
                                          validation_split=0.2,
                                          verbose=0, callbacks=[early_stop])
        else:
            self.history = self.model.fit(self.train_data, 
                                          self.train_labels,
                                          batch_size=batch_size,
                                          epochs=no_epochs,
                                          validation_split=0.2, verbose=0)

    def evaluateModel(self):
        """Evalutes keras ann model against test data"""

        [loss, val] = self.model.evaluate(self.test_data, self.test_labels, 
        verbose=0)
        rmse = np.sqrt(loss)
#        print('ANN RMSE:', rmse) #debug
        return rmse

    def predictWithModel(self, load_profile, plot_results=False):
        """Makes predictions by applying learned ANN model on test data"""

        test_predictions = self.model.predict(load_profile)

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
            self.plotHistory()
            
        return test_predictions

    def plotHistory(self, savefig=False):
        """Plot learning curve"""
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Root Mean Square Error')
        plt.title('ANN Training and Validation Loss Versus Epochs')
        plt.plot(self.history.epoch, 
                 np.array(self.history.history['loss']),
                 label='Training Loss')
        plt.plot(self.history.epoch, 
                 np.array(self.history.history['val_loss']),
                 label='Validation loss')
        plt.legend()
        plt.ylim([0, 0.2])
        if savefig:
            plt.savefig('./data/print/analysis_trainandvalidationloss.pdf')
        plt.show()

    def saveModel(self, name='annmodel'):
        """Save learned ANN model"""
        
        ## TO DO: Test this function
        path = Path(os.path.dirname(__file__)) / 'data/function_mappings'
        model_json = self.model.to_json()
        with open(str(path / (name + ".json")), "w") as file:
            json.dump(model_json, file)
        file.close()

        # serialize weights to HDF5
        self.model.save_weights(str(path / (name + ".h5")))

    def loadModel(self, model_name):
        """Decodes a JSON file into a keras model"""

        path = Path(os.path.dirname(__file__)) / 'data/function_mappings'
        try:
            with (path / (model_name + '.json')).open('r') as ann_model_json:
                model_json_string = ann_model_json.read().\
                replace('\\', '')[1:-1]
                model = model_from_json(model_json_string)
            ann_model_json.close()
            model.load_weights(str(path / (model_name + '.h5')), by_name=False)
            self.model = model
            print('Opening ANN-derived look up table')
            return model
        except BaseException as ex:
            print('Line {} - lookup table loading failed. {}'.format(
                sys.exc_info()[2].tb_lineno, ex))
            return False


class ParametricRegression():
    """Trains linear parametric algorithm using power system load flow load/
       voltage data
    """

    def __init__(self, load_profile=None, voltage_profile=None,
                 train_percentage=0.7, save_model=False):
        
        if load_profile is not None and voltage_profile is not None:
            _split_index = int(train_percentage * len(load_profile))
            self.train = {'data': load_profile[0:_split_index], 
                          'target': voltage_profile[0:_split_index]}
            m = self.train['target'].shape[0]
            self.train['data_pb'] = np.c_[np.ones((m, 1)), self.train['data']]
            # pb stands for plus bias
            self.test = {'data': load_profile[_split_index+1:], 
                         'target': voltage_profile[_split_index+1:]}
            self.calculateTheta()
            self.calculateBias()
            if self.evaluateModel() > 1:
                print('parametric regression rmse too high. Run ANN')

            if save_model:
                self.model_name = 'para_model_' + \
                str(datetime.datetime.now()).\
                replace(':', '-').replace(' ', '_')
                self.saveModel(self.model_name)

    def calculateTheta(self):
        x = tf.constant(self.train['data_pb'], dtype=tf.float32, name="x")
        y = tf.constant(self.train['target'], dtype=tf.float32, name="y")
        xt = tf.transpose(x)
        theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt, x)), xt),
                          y)
        with tf.Session(): ## do I even need this??
            self.theta = theta.eval()

    def calculateBias(self):
        self.bias = [self.theta[0] for i in range(self.theta.shape[1])]

    def predictWithModel(self, load_profile):
        adjusted_bias = []
        for i in range(load_profile.shape[0]):
            adjusted_bias.append(self.bias[0])
        return np.dot(load_profile, self.theta[1:].T) + adjusted_bias

    def evaluateModel(self, plot_results=True):
        prediction = self.predictWithModel(self.test['data'])
        rmse = np.sqrt(
                np.mean((prediction.T[1:].T - self.test['target'].T[1:].T)**2))
#        print('normal equation RMSE: {}'.format(rmse)) #debug
        return rmse

    def saveModel(self, model_name):
        """Encodes parametric model parameters into HDF5 binary data format"""
        
        path = Path(os.path.dirname(__file__)) / 'data/function_mappings/'
        with h5py.File(str(path / (model_name + '.h5')), 'w') as file:
            file.create_dataset(name='data', data=np.array(self.theta))
        file.close()

    def loadModel(self, model_name):
        """Decodes HDF5 binary data format into parametric model parameters"""
    
        path = Path(os.path.dirname(__file__)) / 'data/function_mappings/'
        print(path / (model_name + '.h5'))
        try:
            self.theta = h5py.File(str(path / (model_name + '.h5')), 
                                   'r')['data']
            self.calculateBias()
            print('Opening NE-derived function map')
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
    

class HyperparamSearch():
    """Performs artificial neural network training using various parameters. 
       Returns the rmse of each test
    """
    
    def __init__(self, load_profile=None, voltage_profile=None, 
                 search_type='grid'):
        """Search_type is 'grid' or 'random'..."""
        
        self.solution_matrix = []
        
        if load_profile is not None and voltage_profile is not None:
            start = time()
            tg = {} # tg is short for training grid
            if search_type == 'grid':
                tg['lr'] = np.arange(0.001,0.1,0.02)
                tg['no_hidden_layers'] = np.arange(1,2,1)
                tg['layer_density'] = np.arange(50,60,10)
                self.params = [len(tg['lr']), 
                                   len(tg['no_hidden_layers']),
                                   len(tg['layer_density'])]
                self.solution_matrix.append(self.params)
                #TODO: tg['dropout'] = np.array([False, True])
                #TODO: tg['early_stop'] = np.array([False, True])
                for i in range(len(tg['lr'])):
                    for j in range(len(tg['no_hidden_layers'])):
                        for k in range(len(tg['layer_density'])):
                            ann = ANNRegression(
                                    load_profile, 
                                    voltage_profile, 
                                    lr=tg['lr'][i],
                                    no_hidden_layers=tg['no_hidden_layers'][j],
                                    layer_density=tg['layer_density'][k]
                                    )
                            self.solution_matrix.append(
                                    [tg['lr'][i],
                                    tg['no_hidden_layers'][j],
                                    tg['layer_density'][k],
                                    ann.evaluateModel()[0],
                                    ann.evaluateModel()[1]])
                            print('alpha: {}, layers: {}, density: {}'.format(
                                    tg['lr'][i],
                                    tg['no_hidden_layers'][j],
                                    tg['layer_density'][k]))
                print(self.solution_matrix) #debug
                end = time()
                self.runtime = int(end - start)
                self.exportResults()
                
    
    def exportResults(self):
        """Saves results to file"""
        
        counter = 0
        with open('./data/hyperparamsearchresults_' + str(self.runtime) + 
                  'seconds' + '.txt', 'w') as file:
            for training_session in self.solution_matrix:
                for listitem in training_session:
                    file.write(str(listitem))
                    if counter < len(training_session)-1:
                        file.write(', ')
                    counter += 1
                file.write('\n')
                counter = 0
        file.close()
        
    def importResults(self, runtime):
        """Imports hyperparameter search results file.
           :type: runtime: String
        """
        
        with open('./data/hyperparamsearchresults_' + runtime + 'seconds.txt', 
                  'r') as file:
            results = []
            for line in file:
                results.append([float(i) for i in line.replace('\n', '').
                                split(', ')])
            self.solution_matrix = results
        
    def learningRateAnalysis(self, savefig=None):
        """Plots that demonstrate the effect of alpha as the independent var.
           :type: savefig: index of plot you want to save
        """
        
        lr = []
        self.params = [int(x) for x in self.solution_matrix[0]]
        self.solution_matrix.pop(0)
        for i in range(self.params[1]*self.params[2]):
            for j in range(i, len(self.solution_matrix), 
                           self.params[1]*self.params[2]):
                lr.append(self.solution_matrix[j])
            lr = np.array(lr).T
            plt.plot(lr[0], lr[3], 'r-',
                     lr[0], lr[4], 'b-')
            plt.legend(['MSE', 'MAE'])
            plt.ylabel('MSE/MAE Magnitude')
            plt.xlabel('Learning Rate')
            plt.title('ANN Validation Accuracy Versus Learning Rate for ISH')
            if savefig == i:
                plt.savefig('./data/print/analysis_learningrate.pdf')
            plt.show()
            lr = []
        

def epochAnalysis(study_sizes=[5, 10, 15]):
    """Returns the number of epochs required to meet Keras patience 
       requirements for power flow studies of various sizes.
    """
    #TODO: some weird behaviour. Results seem dependent on size of input array
    
    import sys
    sys.path.append('../')
    from simhandler.powerflowsim import PowerFlowSim
    
    results = []
    for size in study_sizes:
        pfs = PowerFlowSim(50, './data/montecarlo' + str(size) + '.json')
        ann = ANNRegression(pfs.node_loads, pfs.node_voltages, no_epochs=3000,
                       early_stop=True)
        print('pfs and ann training for size {} complete'.format(size))
        results.append([size, ann.history.epoch[-1]])
    print(results)
    return pfs, ann