"""
Intelligent simulation handler (ISH) code by Mike Bardwell, MSc
University of Alberta, 2018
"""

import os
import sys
from powersystemnetwork import Network
from powerflowsim import PowerFlowSim
from regressiontools import TrainANN, NormalEquation
from keras.models import model_from_json

class IntelligentSimulationHandler():
    """Decides on whether to use look-up table or run a new power system
       load flow simulation
    """

    def __init__(self, json_config):
        """Self.network is the network to be simulated and is the point
           of comparison for all of the methods in this class
        """
        
        self.json_config = json_config
        self.network = Network(json_config, False)
        self.compatible_network = None
        self.config_files = self.gatherJsonFiles()
        self.comparisonTests()

    def gatherJsonFiles(self):
        """Searches data folder for json files"""

        config_files = []
        for root, _, files in os.walk('./data/'):
            for file in files:
                if file.endswith('.json') and not file.startswith('ann'):
                    config_files.append(root + file)
        return config_files
    
    def loadLookupTable(self, model_name):
        try:
            with open(model_name + '.json', 'r') as ann_model_json:
                model_json_string = ann_model_json.read().\
                replace('\\', '')[1:-1]
                model = model_from_json(model_json_string)
            ann_model_json.close()
            model.load_weights(model_name + '.h5', by_name=False)
        except BaseException as ex:
            print('Line {} - lookup table loading failed. {}'.format(
                    sys.exc_info()[2].tb_lineno, ex))
            return False
        print('Opening compatible look up table')

    def compareConnections(self, x, threshold=0.95):
        """Compares power system network connections"""

        try:
            x = x['connections']
            y = self.network.config['connections']
        except:
            return False

        bigger_len = len(x) if len(x) > len(y) else len(y)
        smaller_len = len(x) if bigger_len == len(y) else len(y)
        no_similarities = 0
        no_possible_connections = (len(x)*(len(x)-1))/2
        for i in range(smaller_len):
            for j in range(i+1, smaller_len):
                if x[i][j] == y[i][j]:
                    no_similarities += 1
        if (no_similarities/no_possible_connections) < threshold:
            return False
        else:
            return True
    
    def compareGenLoadStorage(self, x, threshold=0.95):
        """Compares power system network loads. Assumes topologies from 
           different json_config files are ordered in the same way.
        """
        
        y = self.network.config['profiles']
        x_len = len(x)
        y_len = len(y)
        smaller_len = x_len if x_len < y_len else y_len
        bigger_len = y_len if smaller_len == x_len else x_len
        no_similarities = 0
        for i in range(smaller_len):
            if x[i] == y[i]:
                no_similarities += 1
        if (no_similarities/bigger_len) < threshold:
            return False
        else:
            return True
            
    def comparisonTests(self):
        """Compares json files in data folder to inputted json config file"""

        for file in self.config_files.copy():
            x = Network(file, False).config
            if (x['lookup_table'] is not False and self.compareConnections(x) 
            and self.compareGenLoadStorage(x['profiles'])):
                self.compatible_network = file
                if self.loadLookupTable('./data/lookup_tables/' 
                                     + x['lookup_table']) == False:
                    self.compatible_network = False

        if self.compatible_network is None:
            print('No compatible look up table found. Running simulation')
            pfs = PowerFlowSim(100, self.json_config)
            #TODO: Add in if len(x['profiles']) condition for ann vs ne
            ann = TrainANN(pfs.node_loads, pfs.node_voltages, save_model=True)
            self.network.saveConfig(ann.model_name)
                    
            

ish = IntelligentSimulationHandler('./data/3node.json')