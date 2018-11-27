"""
Power system load flow (PSLF) simulation code by Mike Bardwell, MSc
University of Alberta, 2018
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import pypsa
import powersystemnetwork as psn

sys.path.append('../')
from simulator import Simulation # imports from exchange library

logging.basicConfig(level=logging.ERROR) # must be called before pypsa
pypsa.pf.logger.setLevel(logging.ERROR) # Use ERROR or DEBUG




class PowerFlowSim():
    """Builds PyPSA-based power system load flow simulation."""

    def __init__(self, length, topology='web',
                 subpath='../_configs/3node.json'):
        """
        type: length: int. Length of usable load profile
        type: topology: String. System network topology (default 'web')
        type: subpath: String. Path to JSON file holding network params
        (default '../_configs/3node.json')
        """

        filename = os.path.join(os.path.dirname(__file__), subpath)
        self.length = length
        self.topology = topology
        self.network_param = psn.Network(filename)
        self.network_param_old = Simulation(filename)
        self.sampleperiod = 60 # default for all files in use
        self.pypsa_network = pypsa.Network(sort=False) # initialise network
        self.pypsa_network.set_snapshots(range(self.__adjustLengths(length)))

        self.nodeloads = np.array([])
        self.nodevoltages = np.array([])

    def noAgents(self):
        """Determines number of network participants."""

        return len(self.network_param_old.participants)

    def attributeCheck(self):
        """TO DO: Implement storage."""

        for agent in self.network_param_old.participants:
            if agent.devices['storage'] is not None:
                print('Agent', agent.id, 'has storage')

    def setupBuses(self):
        """Adds buses to network.

        Important: MW and EUR are default units
        """

        for agent in self.network_param_old.participants:
            self.network.add("Bus", agent.id, v_nom=20.)

    def setupGenerators(self):
        """Adds generators to network."""

        for agent in self.network_param_old.participants:
            if agent.devices['generation'] is None \
               or agent.devices['generation'] == []:
                continue
            else:
                self.network.add(
                    "Generator", "{}-{}".format(agent.id, "Gas"),
                    bus=agent.id,
                    p_nom=0.005, # Nominal household power is 5 kW
                    marginal_cost=0, # REMOVE?
                    p_max_pu=agent.devices['generation']. \
                               offline_profile['profile'],
                    )

    def setupLoads(self):
        """Adds loads to network"""

        for agent in self.network_param_old.participants:
            if agent.devices['load'] is None:
                continue
            else:
                self.network.add("Load",
                                 "{}-Load".format(agent.id),
                                 bus=agent.id,
                                 p_set=agent.devices['load']. \
                                 offline_profile['profile'])

    def setupLines(self, xlines=0.1, rlines=0.1):
        """Adds power lines to network"""

        usedagentlist = []
        if self.topology == 'web':
            for agent in self.network_param_old.participants:
                for otheragent in self.network_param_old.participants:
                    if ((otheragent.id != agent.id) and not (np.any(
                            [otheragent.id == x for x in usedagentlist]))):
                        self.network.add(
                            "Line",
                            "{}-{}".format(agent.id, otheragent.id),
                            bus0=agent.id,
                            bus1=otheragent.id,
                            x=xlines,
                            r=rlines)
                    usedagentlist.append(agent.id)
        elif self.topology == 'ring':
            for agent in self.network_param_old.participants:
                if usedagentlist == []:
                    self.network.add(
                        "Line", "{}-{}".
                        format(self.network_param_old.participants[0].id,
                               self.network_param_old.participants[-1].id),
                        bus0=self.network_param_old.participants[0].id,
                        bus1=self.network_param_old.participants[-1].id,
                        x=xlines,
                        r=rlines)
                    usedagentlist.append(agent.id)
                    continue
                else:
                    self.network.add(
                        "Line",
                        "{}-{}".format(agent.id, usedagentlist[-1]),
                        bus0=agent.id,
                        bus1=usedagentlist[-1],
                        x=xlines,
                        r=rlines)
                    usedagentlist.append(agent.id)
        elif self.topology == 'radial':
            agent = self.network_param_old.participants.copy() # short var
            add = self.network.add
            j = 1
            for i in range(round(len(self.network_param_old.participants)/3)):
                for _ in range(3):
                    if j != len(self.network_param_old.participants):
                        add("Line",
                            "{}-{}".format(agent[i].id,
                                           agent[j].id),
                            bus0=agent[i].id,
                            bus1=agent[j].id,
                            x=xlines,
                            r=rlines)
                        j += 1

    def lengthCap(self):
        """Trims length of load and generation profiles."""

        min_length = float("inf")
        for agent in self.network_param_old.participants:
            if agent.devices['load'] is not None:
                if len(agent.devices['load']. \
                offline_profile['profile']) < min_length:
                    min_length = len(agent.devices['load']. \
                    offline_profile['profile'])
            if agent.devices['generation'] is not None:
                if len(agent.devices['generation']. \
                offline_profile['profile']) < min_length:
                    min_length = len(agent.devices['generation']. \
                    offline_profile['profile'])
        self.__adjustLengths(min_length)
        return min_length

    def __adjustLengths(self, length):
        for agent in self.network_param_old.participants:
            if agent.devices['load'] is not None:
                agent.devices['load'].offline_profile['profile'] = \
                agent.devices['load']. \
                offline_profile['profile'][0:length]
            if agent.devices['generation'] is not None:
                agent.devices['generation'].offline_profile['profile'] \
                = agent.devices['generation']. \
                offline_profile['profile'][0:length]
        return length

    def plotLoads(self, save=False, style='-'):
        """Plots load profiles."""

        plot = []
        for name in self.network.loads_t.p.keys():
            plot.extend(plt.plot(self.network.loads_t.p[name],
                                 style, label=name))
        plt.legend(handles=plot)
        plt.title('Loading Data Over Time')
        plt.xlabel('Iteration (in ' + str(self.sampleperiod)
                   + ' s intervals)')
        plt.ylabel('Load (p.u.)')
        if save:
            plt.savefig(fname='load.pdf', format='pdf')
        plt.show()

    def plotGenerators(self, style='-'):
        """Plots generator profiles."""

        plot = []
        for name in self.network.generators_t.p.keys():
            plot.extend(plt.plot(self.network.generators_t.p[name],
                                 style, label=name))
        plt.legend(handles=plot)
        plt.title('Generation Power Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Generation (p.u.)')
        plt.show()

    def plotNodeVoltages(self, node=None, no_timestamps=None, style='-'):
        """Plots voltage profiles."""

        plot = []
        if no_timestamps is None:
            timestamps = self.length
        else:
            timestamps = no_timestamps
        if node:
            name = self.network.buses_t['v_mag_pu'].keys()[node]
            plotdata = plt.plot(self.network.buses_t.
                                v_mag_pu[name][0:timestamps])
            plt.xlabel('Timestamps')
            plt.ylabel('Node Voltages')
            plt.title('blah')
            plt.show()
            return plotdata[0].get_ydata(True)
        else:
            for name in self.network.buses_t['v_mag_pu'].keys():
                plot.extend(plt.plot(
                    self.network.buses_t.v_mag_pu[name],
                    style, label=name))
            plt.legend(handles=plot)
            plt.title('Node Voltages Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Node Voltages (p.u.)')
            plt.show()
            return None

    def plotLineLoading(self, style='-'):
        """Plots power line loading."""

        plot = []
        for name in self.network.lines_t.p0.keys():
            plot.extend(plt.plot(self.network.lines_t.p0[name],
                                 style, label=name))
        plt.legend(handles=plot)
        plt.title('Line Loading Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Power (p.u.)')
        plt.show()

    def boxPlots(self):
        """Plots voltage magnitudes per node in box plot form"""

        data, toremove = [], []
        keys = self.network.buses_t['v_mag_pu'].keys().tolist()
        for name in keys:
            if self.network.buses['control'][name] == 'Slack':
                toremove.append(name)
                continue
            data.append(self.network.buses_t.v_mag_pu[name])
        [keys.remove(x) for x in toremove]
        plt.subplot(121)
        plt.boxplot(data, labels=keys)
        plt.xticks(rotation=60)
        plt.title('Bus Voltage Box Plots')
        plt.xlabel('Buses')
        plt.ylabel('Voltage (p.u.)')

        ax = plt.subplot(122)
        ax.set_xticks(np.arange(len(keys)) + 1)
        ax.set_xticklabels(keys, rotation=60)
        plt.title('Bus Voltage Box Plots')
        plt.xlabel('Buses')
        plt.ylabel('Voltage (p.u.)')
        plt.violinplot(data)
        plt.show()

    def toArrays(self, load=False, voltage=False):
        """Create numpy arrays with load and voltage profiles"""

        if load:
            for name in self.network.loads_t.p.keys():
                self.nodeloads = np.append(self.nodeloads,
                                           self.network.loads_t.p[name]
                                          )
            self.nodeloads = self.nodeloads.reshape(
                len(self.network.loads_t.p.keys()), self.length).T


        if voltage:
            for name in self.network.buses_t['v_mag_pu'].keys():
                self.nodevoltages = np.append(
                    self.nodevoltages,
                    self.network.buses_t.v_mag_pu[name]
                    )
            self.nodevoltages = self.nodevoltages.reshape(
                len(self.network.buses_t['v_mag_pu'].keys()),
                self.length).T

    def getCurrentTime(self):
        """Access network timestamp"""

        return self.network_param_old.exchange.time()

    def nrPfSim(self, showall=False):
        """Base case for testing power flow class"""
        self.setupBuses()
        self.setupGenerators()
        self.setupLoads()
        self.setupLines()

        self.pypsa_network.pf() # non-linear Newton Raphson power flow

        if showall:
            self.plotLoads()
            self.plotGenerators()
            self.plotNodeVoltages()
            self.plotLineLoading()
            self.boxPlots()

        self.toArrays(True, True)
        