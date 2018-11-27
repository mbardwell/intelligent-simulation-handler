"""
Power system load flow support file by Mike Bardwell, MSc
University of Alberta, 2018
"""

import numpy as np
import sys
import gzip
import pickle
import json

def importZp(filename):
    """Decompress file contents and pipe into pickle object."""
    
    f = gzip.open(filename + '.zp', 'rb')
    p_obj = pickle.load(f)
    f.close()
    return p_obj

def dumpZp(filename, object):
    """Pipe pickled data into compressed file."""
    
    try:
        f = gzip.open(filename + '.zp', 'wb')
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        return True
    except:
        return False

def generateLoadProfile(length = 100, amplitude = 2):
    """Generates loading profile."""
    
    return np.random.rand(int(length))*amplitude

def generateMonteCarloBinaries(filename, starting_no = 0, 
                               no_files = 1, amplitude = 2,
                               computer = 'mikelaptop', writeoverfile = False):
    """Generates Monte Carlo binary files for power system load flow sim.
    
    type: filename: String. Should be the file you want to copy-modify
                    ex: 'C:/Users/mikey/Downloads/zp/18356'
    type: no_files: int. Number of files to create
    type: starting_no: int. First suffix of generated monte carlo file
    type: amplitude: int. Generate rand profile between [0, amplitude]
    type: computer: String. Either 'mikelaptop' or 'mikework'
    """
    from pathlib import Path
    
    if computer == 'mikework':
        sub_dump_path = 'C:/Users/mikey/Downloads/zp/montecarlo'
    else:
        sub_dump_path = 'C:/Users/Michael/Downloads/zp/montecarlo'
    
    network_data = importZp(filename)
    for i in range(no_files):
        dump_path = sub_dump_path + str(starting_no + i)
        if not Path(dump_path + '.zp').is_file() or writeoverfile:
            network_data['load']['profile'] = generateLoadProfile(5e3, 
                                                                  amplitude)
            if starting_no == 0 and i == 0: # only first profile needs gen
                network_data['gen']['profile'] = np.ones(int(5e3))*5
            else:
                network_data['gen']['profile'] = np.array([])
            dumpZp(dump_path, network_data)
    
def viewGenerationProfile(filename):
    """Plots generation profile.
    
    type: filename: String.
    """
    import matplotlib.pyplot as plt
    sim = importZp(filename)
    plt.plot(sim['gen']['profile'])
        

def generateJson(nohouses, computer = 'mikelaptop', auto_proceed = False):
    """Generates JSON file requried for power flow study.
    
    type: nohouses: int. Number of houses in monte carlo study
    type: computer: String. Either 'mikelaptop' or 'mikework'
    type: auto_proceed: boolean. If True, binaries with be auto-created
                        WARNING - many binary files can require GB's of memory
    """
    def fileExists(nohouses):
        """Checks two things. If JSON file and required binaries exist.
        
        type: nohouses: int. Number of houses in monte carlo study
        """
        from pathlib import Path
        
        proceed_flag = auto_proceed
        for i in range(nohouses):
            if computer == 'mikework':
                sub_path = 'C:/Users/mikey/Downloads/zp/'
            elif computer == 'mikelaptop':
                sub_path = 'C:/Users/Michael/Downloads/zp/'
            else:
                print('ERROR: File path not recognized')
                sys.exit(1)
            file_path = Path(sub_path + 'montecarlo' + str(i) + '.zp')
            
            if not file_path.is_file() and not proceed_flag:
                if input('WARN: Will create binaries. Proceed? Y/N ') == 'Y':
                    proceed_flag = True
                else:
                    print('Binaries not available. Process aborted')
                    sys.exit(1)
            if proceed_flag == True:
                _computer = computer # avoids computer = computer below
                generateMonteCarloBinaries((sub_path + '18356'), i,
                                           computer = _computer)

        my_file = Path('../../_configs/montecarlo' + str(nohouses) + '.json')
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
                    "priority" : 2,
                    "logic_type" : "binary",
                    "load": {
                            "scale": 1
                            },
                    "generation": {
                            "available": generation_flag,
                            "scale": {	
                                    "type": "normal",
                                    "scale": 1
                                     }
                                 },
                    "storage": False
                  }
        return profile
        
    if not fileExists(nohouses):
        profiles = []
        for i in range(nohouses):
            if i == 0:
                # First profile should include generation
                profiles.append(generateProfile('montecarlo' + str(i), True))
            else:
                profiles.append(generateProfile('montecarlo' + str(i)))
        
        # start_datetime entry to be replaced with below eventually
        # str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if computer == 'mikework':
            profile_path = "C:/Users/mikey/Downloads/zp/" # work computer
        elif computer == 'mikelaptop':
            profile_path = "C:/Users/Michael/Downloads/zp/" # laptop
        else:
            print('ERROR: File path not recognized')
            exit(0)
        data = {"profile_path" : profile_path, 
                "start_datetime" : '2016-04-01 0:0:0',
                "study": {
                        "description": "binary2-m4y16-super-load"
                         },
                "profiles": profiles
                }
                
        with open('../../_configs/montecarlo' + str(nohouses) + '.json', 'w')\
        as f:
            json.dump(data, f, indent = 2)
            
