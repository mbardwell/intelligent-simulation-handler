![badge pypi](https://img.shields.io/pypi/v/simhandler.svg) ![badge license](https://img.shields.io/pypi/l/simhandler.svg)

# intelligent-simulation-handler

### What is this repository for? ###

* Software development of an intelligent simulation handler for power system load flow simulations
* Project completed as partial requirement of attaining MSc at the University of Alberta  

### How do I get set up? ###

* *pip install simhandler*
* To run the basic 3 node example:
  1. Download [3node.json](https://github.com/mbardwell/intelligent-simulation-handler/tree/master/simhandler/data/network_configurations) file

  *Put it somewhere convenient (ie: Downloads folder) as *3node.json*
    
  2. In a Python editor write the following basic test:
  
'''

import numpy as np

from simhandler.smartsim import SmartPSLF

configuration_file = *insert path to your 3node.json file* (ex: 'C:/Users/mikey/Downloads/3node.json')

sim = SmartPSLF(configuration_file)

fake_load_profile = np.ones((10,3)) # Ten timesteps, three input nodes

print(sim.map.predictWithModel(fake_load_profile)) # Prints out ten voltage timesteps. Three nodes per timestep

'''

  3. It should look something like below![photo of basic test](https://user-images.githubusercontent.com/11367325/50410536-ce69e700-07b6-11e9-979e-51633080eb35.PNG)
     
### Contribution guidelines ###

* Follow PEP8 guidelines with the following exception: use lowercaseUppercase function naming instead of snake_case

### Questions? ###

* If you find a bug or want a feature. Create an issue
* Otherwise, send emails with title 'Intelligent Simulation Handler - *Question/Comment*' to bardwell@ualberta.ca