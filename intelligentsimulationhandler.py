"""
Intelligent simulation handler (ISH) code by Mike Bardwell, MSc
University of Alberta, 2018
"""

import matplotlib.pyplot as plt
import numpy as np


class IntelligentSimulationHandler(object):
    """Runs power system load flow analysis using numerical methods or
       machine learning
    """
    
    def __init__(self):
        