"""
Power system load flow (PSLF) simulation code by Mike Bardwell, MSc
The term node(s) is used interchangeably with participant(s)
University of Alberta, 2018
"""

import sys
sys.path.append('../') # when running code locally, this includes simhandler
                       # package.
import logging
import matplotlib.pyplot as plt
import numpy as np
import pypsa
import simhandler.powersystemnetwork as psn

logging.basicConfig(level=logging.ERROR) # must be called before pypsa
pypsa.pf.logger.setLevel(logging.ERROR) # Use ERROR or DEBUG

class PowerFlowSim():
    """Builds PyPSA-based power system load flow simulation."""

    def __init__(self, length, 
                 filename='./data/network_configurations/3node.json', 
                 showall=False):
        """
        type: length: int. Length of usable load profile
        type: filename: String. Config file storing power system network data
        type: showall: bool. Show plots after running simulation
        """

        self.length = length
        self.network_param = psn.Network(filename)
        self.sampleperiod = 60 # default for all files in use
        self.pypsa_network = pypsa.Network(sort=False) # initialise network
        self.pypsa_network.set_snapshots(range(self._adjustLengths(length)))
        
        self.setupBuses()
        self.setupGenerators()
        self.setupLoads()
        self.setupLines()

        self.pypsa_network.pf() # non-linear Newton Raphson power flow
        self.toArrays(True, True)

        if showall:
            self.plotLoads()
            self.plotGenerators()
            self.plotNodeVoltages()
            self.plotLineLoading()
            self.boxPlots()

    def noAgents(self):
        """Determines number of network participants."""
        return len(self.network_param.participants)

    def setupBuses(self):
        """Adds buses to network.

        Important: MW and EUR are default units
        """

        nodes = self.network_param.participants # shorthanding
        for bus in nodes:
            self.pypsa_network.add("Bus", bus, v_nom=20.)

    def setupGenerators(self):
        """Adds generators to network."""

        nodes = self.network_param.participants # shorthanding
        for bus in nodes:
            if len(nodes[bus].gen['profile']) == 0:
                continue
            else:
                self.pypsa_network.add(
                    "Generator", "{}-{}".format(bus, "Gas"),
                    bus=bus,
                    p_nom=0.005, # Nominal household power is 5 kW
                    marginal_cost=0, # REMOVE?
                    p_max_pu=nodes[bus].gen['profile']
                    )

    def setupLoads(self):
        """Adds loads to network"""

        nodes = self.network_param.participants # shorthanding
        for bus in nodes:
            if nodes[bus].load is None:
                continue
            else:
                self.pypsa_network.add("Load",
                                       "{}-Load".format(bus),
                                       bus=bus,
                                       p_set=nodes[bus].load['profile']
                                       )

    def setupLines(self, topology='web', x_lines=0.001, r_lines=0.001):
        """Adds power lines to network"""
                    
        def _addLine(bus, other_bus, x_line=x_lines, r_line=r_lines):
            self.pypsa_network.add("Line",
                                   "{}-{}".format(bus, other_bus),
                                   bus0=bus,
                                   bus1=other_bus,
                                   x=x_line,
                                   r=r_line)

        connection_matrix = self.network_param.config['connections']
        node_names = list(self.network_param.participants.keys()) # shorthand
        for i in range(len(connection_matrix)):
            for j in range(i+1, len(connection_matrix)):
                if connection_matrix[i][j] == 1:
                    _addLine(node_names[i], node_names[j])

    def lengthCap(self):
        """Trims length of load and gen profiles to min common length."""

        min_length = float("inf")
        nodes = self.network_param.participants # shorthanding
        for bus in nodes:
            if nodes[bus].load is not None:
                if len(nodes[bus].load['profile']) < min_length:
                    min_length = len(nodes[bus].load['profile'])
            if nodes[bus].gen is not None:
                if len(nodes[bus].gen['profile']) < min_length:
                    min_length = len(nodes[bus].gen['profile'])
        self._adjustLengths(min_length)
        return min_length

    def _adjustLengths(self, length):

        nodes = self.network_param.participants # shorthanding
        for bus in nodes:
            if nodes[bus].load is not None:
                nodes[bus].load['profile'] = \
                nodes[bus].load['profile'][0:length]
            if nodes[bus].gen is not None:
                nodes[bus].gen['profile'] = nodes[bus].gen['profile'][0:length]
        return length

    def plotLoads(self, save=False, style='-'):
        """Plots load profiles."""

        plot = []
        for bus in self.pypsa_network.loads_t.p.keys():
            plot.extend(plt.plot(self.pypsa_network.loads_t.p[bus],
                                 style, label=bus))
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
        for bus in self.pypsa_network.generators_t.p.keys():
            plot.extend(plt.plot(self.pypsa_network.generators_t.p[bus],
                                 style, label=bus))
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
            bus = self.pypsa_network.buses_t['v_mag_pu'].keys()[node]
            plotdata = plt.plot(self.pypsa_network.buses_t.
                                v_mag_pu[bus][0:timestamps])
            plt.xlabel('Timestamps')
            plt.ylabel('Node Voltages')
            plt.title('blah')
            plt.show()
            return plotdata[0].get_ydata(True)
        else:
            for bus in self.pypsa_network.buses_t['v_mag_pu'].keys():
                plot.extend(plt.plot(
                    self.pypsa_network.buses_t.v_mag_pu[bus],
                    style, label=bus))
            plt.legend(handles=plot)
            plt.title('Node Voltages Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Node Voltages (p.u.)')
            plt.show()
            return None

    def plotLineLoading(self, style='-'):
        """Plots power line loading."""

        plot = []
        for bus in self.pypsa_network.lines_t.p0.keys():
            plot.extend(plt.plot(self.pypsa_network.lines_t.p0[bus],
                                 style, label=bus))
        plt.legend(handles=plot)
        plt.title('Line Loading Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Power (p.u.)')
        plt.show()

    def boxPlots(self, save=False):
        """Plots voltage magnitudes per node in box plot form"""

        data, toremove = [], []
        keys = self.pypsa_network.buses_t['v_mag_pu'].keys().tolist()
        for bus in keys:
            if self.pypsa_network.buses['control'][bus] == 'Slack':
                toremove.append(bus)
                continue
            data.append(self.pypsa_network.buses_t.v_mag_pu[bus])
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
        plt.title('Bus Voltage Violin Plots')
        plt.xlabel('Buses')
        plt.ylabel('Voltage (p.u.)')
        plt.violinplot(data)       
        plt.tight_layout()
        if save:
            plt.savefig(fname='boxandviolinplot.pdf', format='pdf')
        plt.show()

    def toArrays(self, load=False, voltage=False):
        """Create numpy arrays with load and voltage profiles"""
        
        self.node_loads = np.array([])
        self.node_voltages = np.array([])
        if load:
            for bus in self.pypsa_network.loads_t.p.keys():
                self.node_loads = np.append(self.node_loads,
                                           self.pypsa_network.loads_t.p[bus]
                                          )
            self.node_loads = self.node_loads.reshape(
                len(self.pypsa_network.loads_t.p.keys()),
                self.length).T


        if voltage:
            for bus in self.pypsa_network.buses_t['v_mag_pu'].keys():
                self.node_voltages = np.append(
                    self.node_voltages,
                    self.pypsa_network.buses_t.v_mag_pu[bus]
                    )
            self.node_voltages = self.node_voltages.reshape(
                len(self.pypsa_network.buses_t['v_mag_pu'].keys()),
                self.length).T