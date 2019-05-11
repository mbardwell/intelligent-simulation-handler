"""
Intelligent simulation handler (ISH) code by Mike Bardwell, MSc
University of Alberta, 2018
"""

import os
from .powersystemnetwork import Network
from .powerflowsim import PowerFlowSim
from .regressiontools import ANNRegression, IdentityLinearRegression
from pathlib import Path
# TODO: from simhandler.datageneration import generateJson


class SmartPSLF():
    """Decides on whether to use look-up table or run a new power system
       load flow simulation
    """

    def __init__(self, json_config, force_use_of_solver=False):
        """type: json_config: String. Path to json config file.
           Self.network is the network to be simulated and is the point
           of comparison for all of the methods in this class
        """

        self.json_config = json_config
        self.network = Network(json_config, False)
        self.map = None
        self.config_files = self.gatherJsonFiles()
        self.comparisonTests(force_use_of_solver)

    def gatherJsonFiles(self):
        """Searches data folder for json files"""

        config_files = []
        path = Path(os.path.dirname(__file__)) / 'data/network_configurations/'
        for root, _, files in os.walk(str(path)):
            for file in files:
                if file.endswith('.json') and not file.startswith('ann'):
                    config_files.append(root + '\\' + file)
        if config_files == []:
            raise Exception('No configuration files found')
        return config_files

    def compareConnections(self, config, threshold=0.95):
        """Compares power system network connections"""

        try:
            x = config['connections']
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
        return True

    def compareGenLoadStorage(self, config_profiles, threshold=0.95):
        """Compares power system network loads. Assumes topologies from
           different json_config files are ordered in the same way.
        """

        x = config_profiles
        y = self.network.config['profiles']
        x_len = len(config_profiles)
        y_len = len(y)
        smaller_len = x_len if x_len < y_len else y_len
        bigger_len = y_len if smaller_len == x_len else x_len
        no_similarities = 0
        for i in range(smaller_len):
            if x[i] == y[i]:
                no_similarities += 1
        if (no_similarities/bigger_len) < threshold:
            return False
        return True

    def comparisonTests(self, force_use_of_solver=False):
        """Compares json files in data folder to inputted json config file"""

        if not force_use_of_solver:
            for file in self.config_files.copy():
                x = Network(file, False).config
                if (x['lookup_table'] is not False and
                        self.compareConnections(x) and
                        self.compareGenLoadStorage(x['profiles'])):
    
                    if len(x['profiles']) > 2:
                        function_map = ANNRegression()
                    else:
                        function_map = IdentityLinearRegression()
                    function_map.loadModel(x['lookup_table'])
                    self.map = function_map
                    print('Compatible network found in file: ', file)
                    break

        if self.map is None or force_use_of_solver:
            print('No compatible look up table found. Deploying PyPSA solver')
            pfs = PowerFlowSim(100, self.json_config)

            network_name = self.json_config.rsplit('/', 1)[-1]
            new_config = Path(os.path.dirname(__file__)) /\
                ('data/network_configurations/' + network_name)
            self.network.json_config = str(new_config)
            if pfs.node_loads.shape[1] > 2:
                function_map = ANNRegression(pfs.node_loads,
                                             pfs.node_voltages,
                                             save_model=True)
            else:
                function_map = IdentityLinearRegression(
                    pfs.node_loads, pfs.node_voltages, save_model=True)
            self.network.saveConfig(function_map.model_name)
            self.map = function_map
