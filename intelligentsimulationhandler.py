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
        self.config_files = self.gatherFiles()
        self.totalComparison()
        
    def gatherFiles(self):
        """Searches data folder for json files"""
        
        config_files = []
        for root, _, files in os.walk('./data/'):
            for file in files:
                if file.endswith('.json'):
                    config_files.append(root + file)
        return config_files
    
    def totalComparison(self):
        """Compares json files in data folder to inputted json config file"""
        
        for file in self.config_files:
            x = Network(file, False).config
            y = self.network.config
            shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
            print(file, len(shared_items))
        
ish = IntelligentSimulationHandler('./data/3node.json')