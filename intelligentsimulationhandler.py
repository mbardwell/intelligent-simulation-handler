"""
Intelligent simulation handler (ISH) code by Mike Bardwell, MSc
University of Alberta, 2018
"""

import os
from powersystemnetwork import Network

class IntelligentSimulationHandler():
    """Decides on whether to use look-up table or run a new power system
       load flow simulation
    """

    def __init__(self, json_config):
        self.network = Network(json_config, False)
        self.config_files = self.gatherJsonFiles()
        self.compareConnections()

    def gatherJsonFiles(self):
        """Searches data folder for json files"""

        config_files = []
        for root, _, files in os.walk('./data/'):
            for file in files:
                if file.endswith('.json'):
                    config_files.append(root + file)
        return config_files

    def compareConnections(self, threshold=0.95):
        """Compares power system network connections"""

        for file in self.config_files.copy():
            try:
                x = Network(file, False).config['connections']
            except:
                self.config_files.remove(file)
                continue
            y = self.network.config['connections']

            bigger_len = len(x) if len(x) > len(y) else len(y)
            smaller_len = len(x) if bigger_len == len(y) else len(y)
            no_similarities = 0
            no_possible_connections = (len(x)*(len(x)-1))/2
            for i in range(smaller_len):
                for j in range(i+1, smaller_len):
                    if x[i][j] == y[i][j]:
                        no_similarities += 1
            if (no_similarities/no_possible_connections) < threshold:
                self.config_files.remove(file)

    def totalComparison(self):
        """Compares json files in data folder to inputted json config file"""

        for file in self.config_files:
            x = Network(file, False).config
            y = self.network.config
            shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
            print(file, len(shared_items))

ish = IntelligentSimulationHandler('./data/3node.json')