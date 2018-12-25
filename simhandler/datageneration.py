"""
Power system load flow support file by Mike Bardwell, MSc
University of Alberta, 2018
"""

import numpy as np
import sys
import gzip
import pickle
import json
from pathlib import Path

def importZp(file_name):
    """Decompress file contents and pipe into pickle object."""
    
    try:
        f = gzip.open(str(file_name) + '.zp', 'rb')
        p_obj = pickle.load(f)
        f.close()
        return p_obj
    except IOError as ex:
        print(ex)

def dumpZp(file, object):
    """Pipe pickled data into compressed file."""
    
    try:
        f = gzip.open(str(file), 'wb')
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except:
        raise Exception('Monte Carlo loadgen profile was not generated.')

def generateLoadProfile(length=100, amplitude=2):
    """Generates loading profile."""
    
    return np.random.rand(int(length))*amplitude

def generateMonteCarloBinaries(mod_file, starting_no=0, no_files=1, 
                               length=20000, amplitude=2, writeoverfile=False):
    """Generates Monte Carlo binary files for power system load flow sim.
    
    type: mod_file: String. Should be the file you want to copy-modify
                    ex: './data/loadgen_profiles/example1'
    type: no_files: int. Number of files to create
    type: starting_no: int. First suffix of generated monte carlo file
    type: amplitude: int. Generate rand profile between [0, amplitude]
    type: filename: String. Default 'montecarlo'
    """
    
    network_data = importZp(mod_file) # imports file with desired dict
    for i in range(no_files):
        dump_path = Path(__file__).parent / 'data/loadgen_profiles'
        file = 'montecarlo' + str(starting_no + i) + '.zp'
        if not (dump_path / file).is_file() or writeoverfile:
            network_data['load']['profile'] = generateLoadProfile(length, 
                                                                  amplitude)
            if starting_no == 0 and i == 0: # only first profile needs gen
                network_data['gen']['profile'] = np.ones(int(length))*5
            else:
                network_data['gen']['profile'] = np.array([])
            dumpZp(dump_path / file, network_data)

def viewGenerationProfile(filename):
    """Plots generation profile.
    
    type: filename: String.
    """
    import matplotlib.pyplot as plt
    sim = importZp(filename)
    plt.plot(sim['gen']['profile'])
        

def generateJson(no_houses, topology='radial', auto_proceed=False):
    """Generates JSON file requried for power flow study.
    
    type: no_houses: int. Number of houses in monte carlo study
    type: auto_proceed: boolean. If True, binaries with be auto-created
                        WARNING - many binary files can require GB's of memory
    """

    def fileExists(no_houses):
        """Checks two things. If JSON file and required binaries exist.
        
        type: no_houses: int. Number of houses in monte carlo study
        """
        
        proceed_flag = auto_proceed
        for i in range(no_houses):
            file_path = Path(__file__).parent /\
            ('data/montecarlo' + str(i) + '.zp')
            
            if not file_path.is_file() and not proceed_flag:
                prompt = input('WARNING: Will create binaries. Proceed? Y/N ')
                if prompt == 'Y':
                    proceed_flag = True
                elif prompt == 'N':
                    print('Binaries not available. Process aborted')
                    sys.exit(1)
                else:
                    print('Not a valid input. Y or N. Process aborted')
                    sys.exit(1)
            if proceed_flag == True:
                generateMonteCarloBinaries(Path(__file__).parent / 
                                           ('data/loadgen_profiles/example1'), 
                                           i)

        my_file = Path(__file__).parent / ('data/montecarlo' + str(no_houses) + '.json')
        if my_file.is_file():
            return True
        else: 
            return False
        
    
    def generateProfile(fileid, generation_flag = False):
        """Generates JSON profile for individual topologies.
        
        type: fileid: String, generation_flag: bool
        """
        profile = {
                    "id" : str(fileid),
                    "logic_type" : "binary",
                    "load": True,
                    "generation": generation_flag,
                    "storage": False
                  }
        return profile
    
    def generateConnections(topology='radial'):
        """Generates connection topology for power system load flow
           montecarlo json_config files
        """
        
        connection_matrix = []
        if topology == 'radial':
            radial_constant = 3
            for i in range(no_houses):
                connection_rows = []
                for j in range(no_houses):
                    if i == j or j < radial_constant*i+1 \
                    or j > radial_constant*(i+1):
                        connection_rows.append(0)
                    else:
                        connection_rows.append(1)
                connection_matrix.append(connection_rows)
        if topology == 'web':
            for i in range(no_houses):
                connection_rows = []
                for j in range(no_houses):
                    if i == j: 
                        connection_rows.append(0)
                    else:
                        connection_rows.append(1)
                connection_matrix.append(connection_rows)
        if topology == 'ring':
            for i in range(no_houses):
                connection_rows = []
                for j in range(no_houses):
                    if (j == i+1 or j == i-1 or j == i+no_houses-1 
                        or j == i-no_houses+1): 
                        connection_rows.append(1)
                    else:
                        connection_rows.append(0)
                connection_matrix.append(connection_rows)
            
        return connection_matrix
    
    if not fileExists(no_houses):
        profiles = []
        for i in range(no_houses):
            if i == 0:
                # First profile should include generation
                profiles.append(generateProfile('montecarlo' + str(i), True))
            else:
                profiles.append(generateProfile('montecarlo' + str(i)))
        
        # start_datetime entry to be replaced with below eventually
        # str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {"profile_path" : './data/loadgen_profiles/', 
                "start_datetime" : '2016-04-01 0:0:0',
                "study": 'TODO: Prompt ISH user for name of study',
                "lookup_table": False,
                "connections": generateConnections(topology),
                "profiles": profiles
                }
        
        file = Path(__file__).parent /\
        ('data/network_configurations/montecarlo' + str(no_houses) + '.json')
        with file.open('w') as f:
            json.dump(data, f, indent = 2)
            
