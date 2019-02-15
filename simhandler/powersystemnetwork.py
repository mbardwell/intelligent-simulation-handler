"""
Power system load flow (PSLF) network classes by Mike Bardwell, MSc
For handling PSLF network data
University of Alberta, 2018
"""

import sys
import gzip
import pickle
import json
import traceback
from pathlib import Path
import random
from simhandler.datageneration import generateMonteCarloBinaries, generateJson


class Network():
    """Builds network for power system load flow simulation."""

    def __init__(self, json_config, include_profiles=True):
        """json_config: absolute path to json file"""

        self.json_config = json_config
        self.config = self.getConfig(json_config)
        self.participants = {}
        if self.config is not None and include_profiles:
            self.addParticipants()

    def getConfig(self, json_config):
        """Imports JSON-based configuration file."""

        if 'montecarlo' in json_config and not Path(json_config).is_file():
            no_features = int(str(json_config).rsplit('montecarlo',1)[-1].
                             rsplit('.json')[0]) 
            print('Generating montecarlo json with {} features'.
                  format(no_features))
            generateJson(no_features)
            
        try:
            with open(json_config, 'r') as data_file:
                config = json.load(data_file)
            data_file.close()
            return config    
        except IOError:
            traceback.print_exc(file=sys.stdout)

            return None            
        
    def saveConfig(self, model_name):
        self.config['lookup_table'] = model_name
        try:
            with open(self.json_config, 'w') as config_file:
                json.dump(self.config, config_file, indent=2)
        except IOError as ex:
            print('File error: ', ex)
            print('Debug: ', self.json_config, Path(__file__))

    def addParticipants(self):
        """Adds all time-based profiles to participant attribute."""
        counter = 0

        for profile in self.config['profiles']:
            current_name = self.generateRandomName()
            if current_name in self.participants.keys():
                current_name = current_name + '_' + str(counter)
                counter += 1
            file_name = profile['id'] + '.zp'
            self.participants[current_name] = Participant(file_name)

    def generateRandomName(self):
        """To be replaced by names from json_config file when updated"""
        
        file_path = Path(__file__).parent
        file = file_path / ('utils/' + 'us_census_male_names.txt')

        with open(str(file), 'r') as namefile:
            names = namefile.read().splitlines()
        namefile.close()
        return random.choice(names)


class Participant():
    """Participants handle time-based profiles of nodes in power system load
       flow sims. Generation will be shorthanded as gen throughout this class.
    """

    def __init__(self, file_name):
        """self.gen/load keys: 'interval_s', 'units', 'profile', 'start_time'.
        """

        self.profiles = self.importTimeProfiles(file_name)
        self.gen= None
        self.load = None

        if self.profiles is not None:
            self.gen = self.profiles['gen']
            self.load = self.profiles['load']

    def importTimeProfiles(self, file_name):
        """Decompress file contents and pipe into pickle object."""
        
        file = Path(__file__).parent / ('data/loadgen_profiles/' + file_name)
        try:
            f = gzip.open(str(file), 'rb')
            profile = pickle.load(f)
            f.close()
            return profile
        except:
            try:
                generateMonteCarloBinaries(Path(__file__).parent /\
                                           'data/loadgen_profiles/example1.zp')
            except:
                raise Exception('loadgen profiles not loaded')

    def dumpTimeProfiles(self, file_name):
        """Pipe pickled data into compressed file."""

        try:
            f = gzip.open(file_name, 'wb')
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

