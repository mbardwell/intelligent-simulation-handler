"""
Power system load flow (PSLF) network classes by Mike Bardwell, MSc
For handling PSLF network data
University of Alberta, 2018
"""

import gzip
import pickle
import json

class Network():
    """Builds network for power system load flow simulation."""

    def __init__(self, json_config, include_profiles=True):
        self.json_config = json_config
        self.config = self.getConfig(json_config)
        self.participants = {}
        if self.config is not None and include_profiles:
            self.addParticipants()

    def getConfig(self, json_config):
        """Imports JSON-based configuration file."""

        try:
            with open(json_config, 'r') as data_file:
                config = json.load(data_file)
            return config
        except IOError as ex:
            print('Exception: ', ex)
            return None
        
    def saveConfig(self, model_name):
        self.config['lookup_table'] = model_name
        with open(self.json_config, 'w') as config_file:
            json.dump(self.config, config_file, indent = 2)

    def addParticipants(self):
        """Adds all time-based profiles to participant attribute."""

        for profile in self.config['profiles']:
            current_name = self.generateRandomName()
            self.participants[current_name] = None
            if profile['id'] in self.participants.keys():
                print('Error: profile already exists for id: ', profile['id'])
            else:
                filename = profile['id'] + '.zp'
                self.participants[current_name] = Participant(filename)

    def generateRandomName(self):
        """To be replaced by names from json_config file when updated"""

        import random
        with open('./utils/us_census_male_names.txt', 'r') as namefile:
            names = namefile.read().splitlines()
        namefile.close()
        return random.choice(names)

class Participant():
    """Participants handle time-based profiles of nodes in power system load
       flow sims. Generation will be shorthanded as gen throughout this class.
    """

    def __init__(self, filename):
        """self.gen/load keys: 'interval_s', 'units', 'profile', 'start_time'.
        """

        self.profiles = self.importTimeProfiles(filename)
        self.gen= None
        self.load = None
        
        if self.profiles is not None:
            self.gen = self.profiles['gen']
            self.load = self.profiles['load']

    def importTimeProfiles(self, filename):
        """Decompress file contents and pipe into pickle object."""
        
        path = './data/'
        try:
            f = gzip.open(path + filename, 'rb')
            profile = pickle.load(f)
            f.close()
            return profile
        except IOError as ex:
            print('Exception: ', ex)
            return None

    def dumpTimeProfiles(self, filename):
        """Pipe pickled data into compressed file."""

        try:
            f = gzip.open(filename, 'wb')
            self.updateProfiles()
            pickle.dump(self.profiles,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            return True
        except IOError as ex:
            print('Exception: ', ex)
            return None

    def updateProfiles(self):
        """Update profiles attribute with current gen/load profiles."""

        self.profiles = {'gen': self.gen, 'load': self.load}

    def addToGeneration(self):
        """TODO: Adds generation to current generation profile."""

        print(1)

    def addToLoad(self):
        """TODO: Adds load to current load profile."""

        print(2)