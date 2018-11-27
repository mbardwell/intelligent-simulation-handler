# -*- coding: utf-8 -*-
"""
Linear regression tools for Power Flow emulation

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
import numpy as np

class TrainANN(object):
    def __init__(self, load_profile, voltage_profile, train_percentage = 0.7, name = 'ann_model'):
        """
        :type load_profile: List[int], voltage_profile: List[int]
        :type train_percentage: int, name: String
        """
        _split_index = int(train_percentage * len(load_profile))
        self.train_data = load_profile[0:_split_index]
        self.train_labels = voltage_profile[0:_split_index]
        
        self.test_data = load_profile[_split_index+1:]
        self.test_labels = voltage_profile[_split_index+1:]
        
        self.name = name
        
    def buildModel(self):
        """
        :rtype self.model: class 'tensorflow.python.keras.engine.sequential.Sequential'
        """
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(64, 
                                     activation=tf.nn.relu, 
                                     input_shape=(self.train_data.shape[1],)))
#        self.model.add(keras.layers.Dropout(0.2))
#        self.model.add(keras.layers.Dense(64, activation=tf.nn.relu))
#        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(self.train_labels.shape[1]))
    
        optimizer = tf.train.RMSPropOptimizer(0.001)
    
        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['mae'])
#        self.model.summary() # for debug
    
    def trainModel(self, no_epochs = 1000):
        """
        :type model: class 'tensorflow.python.keras.engine.sequential.Sequential'
        :type epochs: int
        :rtype history: ??
        """
        
        # The patience parameter is the amount of epochs to check for improvement.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=200)
            
        # Store training stats
        self.history = self.model.fit(self.train_data, self.train_labels, epochs=no_epochs,
                                      validation_split=0.2, verbose=0,
                                      callbacks=[early_stop])

    def evaluateModel(self):
        [loss, mae] = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
        print("ANN regression loss: {}, mae: {}".format(loss, mae))
        return mae
    
    def predictWithModel(self, plot_results = True):
        test_predictions = self.model.predict(self.test_data)

        if plot_results:
            plt.plot(self.test_labels, test_predictions, 'o')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.axis('equal')
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.plot([-100, 100],[-100,100])
            plt.show()
            
            # Histogram
            error = test_predictions - self.test_labels
            
            for i in range(len(error[0])):
                plt.hist(error.T[i], bins = 50)
            plt.xlabel("Prediction Error")
            plt.ylabel("Count")
            plt.show()
            
    def saveModel(self, name = 'annmodel'):
        ## TO DO: Test this function
        # serialize model to JSON
        model_json = self.model.to_json()
        with open('../testing/' + name + ".json", "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        self.model.save_weights('../testing/' + name + ".h5")
        print("Saved model to disk")

    def plotHistory(self):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(self.history.epoch, np.array(self.history.history['mean_absolute_error']), 
                 label='Train Loss')
        plt.plot(self.history.epoch, np.array(self.history.history['val_mean_absolute_error']),
                 label = 'Val loss')
        plt.legend()
        plt.ylim([0,0.2])
        plt.show()
        
class TrainRNN(object):
    def __init__(self, load_profile, voltage_profile, train_percentage = 0.7, name = 'ann_model'):
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


class NormalEquation(object):
    def __init__(self, load_profile, voltage_profile):
        self.lpf = {'data': load_profile, 
                    'target': voltage_profile}
        mlpf = self.lpf['target'].shape[0]
        self.lpf['data_pb'] = np.c_[np.ones((mlpf, 1)), self.lpf['data']]
        
    def calculateTheta(self):
        xlpf = tf.constant(self.lpf['data_pb'], 
                           dtype = tf.float32, name = "xlpf")
        ylpf = tf.constant(self.lpf['target'], 
                           dtype = tf.float32, name = "ylpf") 
        xlpft = tf.transpose(xlpf)
        thetalpf = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xlpft, xlpf)), xlpft), ylpf)
        with tf.Session(): ## do I even need this??
            self.thetalpf_value = thetalpf.eval()
            
    def calculateBias(self):
        biaslpf = []
        for i in range(self.lpf['target'].shape[0]):
            biaslpf.append(self.thetalpf_value[0])
        self.biaslpf = np.array(biaslpf)
        
    def predictWithModel(self, load_profile):
        return np.dot(load_profile, self.thetalpf_value[1:].T) + self.biaslpf
        
    def evaluateModel(self, plot_results = True):
        prediction = self.predictWithModel(self.lpf['data'])
        mselpf = np.mean((prediction.T[1:].T - self.lpf['target'].T[1:].T)**2)
        maelpf = abs(np.mean((prediction.T[1:].T - self.lpf['target'].T[1:].T)))
        print('normal equation mse: {}, mae: {}'.format(mselpf, maelpf))
        return maelpf