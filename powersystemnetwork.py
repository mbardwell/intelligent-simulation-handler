"""
Power system load flow network classes by Mike Bardwell, MSc
For use in power system load flow applications
University of Alberta, 2018
"""

import gzip
import pickle
import json


class Network():
    """Builds network for power system load flow simulation."""

    def __init__(self, json_config):
        self.config = self.getConfig(json_config)
        if self.config is not None:
            self.addParticipants()

    def getConfig(self, json_config):
        """Imports JSON-based configuration file."""

        try:
            with open(json_config) as data_file:
                config = json.load(data_file)
            return config
        except IOError as ex:
            print('Exception: ', ex)
            return None

    def addParticipants(self):
        """Adds all time-based profiles to participant attribute."""

        self.participant = {}
        for profile in self.config['profiles']:
            current_name = self.generateRandomName()
            self.participant[current_name] = None
            if profile['id'] in self.participant.keys():
                print('Error: profile already exists for id: ', profile['id'])
            else:
                filename = profile['id'] + '.zp'
                self.participant[current_name] = Participant(filename)

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
        if self.profiles is not None:
            self.gen = self.profiles['gen']
            self.load = self.profiles['load']

    def importTimeProfiles(self, filename):
        """Decompress file contents and pipe into pickle object."""

        try:
            f = gzip.open(filename, 'rb')
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
        """Update profiles attribute with current gen/load profiles"""

        self.profiles = {'gen': self.gen, 'load': self.load}

    def addToGeneration(self):
        """Adds generation to current generation profile"""

        print(1)

    def addToLoad(self):
        """Adds load to current load profile"""

        print(2)
        