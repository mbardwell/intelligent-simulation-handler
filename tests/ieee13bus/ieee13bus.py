## Example of transformer with non-trivial phase shift and tap ratio
#
#This example is a copy of pandapower's minimal example.
import logging
logging.basicConfig(level=logging.ERROR)

import pypsa
pypsa.pf.logger.setLevel(logging.ERROR)
import numpy as np

network = pypsa.Network()

network.add("Bus","650",v_nom=115,v_mag_pu_set=1.00)
network.add("Bus","632",v_nom=4.16)
network.add("Bus","633",v_nom=4.16)
network.add("Bus","634",v_nom=0.48)
network.add("Bus","645",v_nom=4.16)
network.add("Bus","646",v_nom=4.16)
network.add("Bus","671",v_nom=4.16)
network.add("Bus","692",v_nom=4.16)
network.add("Bus","675",v_nom=4.16)
network.add("Bus","680",v_nom=4.16)
network.add("Bus","684",v_nom=4.16)
network.add("Bus","652",v_nom=4.16)
network.add("Bus","611",v_nom=4.16)

#network.add("Transformer","MV-LV trafo",type="25 MVA 110/10 kV",bus0="MV2 bus",bus1="LV1 bus")
network.add("Transformer","1",
            x = 0.08,
            r = 0.01,
            s_nom = 5, # in MVA
            s_max_pu = 1,
            bus0="650",
            bus1="632"
            )
network.add("Transformer","2",
            x = 0.02,
            r = 0.011,
            s_nom = 0.5, # in MVA
            s_max_pu = 1,
            bus0="633",
            bus1="634"
            )

# 2000 ft = 0.61 km, 2500 ft = 0.762 km
# since we are calculating delta-delta, use phase cable impedance
# 0.306 ohm/mile = 0.191 ohm/km, 0.721 inch = 18.3 mm -> 263 mm^2 cross sect

network.add("Line","2",
            x = 0.001,
            r = 0.191*0.762,
            bus0="632",bus1="633",
            length=0.1525)
network.add("Line","3",
            x = 0.001,
            r = 0.191*0.762,
            bus0="632",bus1="645",
            length=0.1525)
network.add("Line","4",
            x = 0.001,
            r = 0.191*0.762,
            bus0="645",bus1="646",
            length=0.0915)
network.add("Line","5",
            x = 0.001,
            r = 0.191*0.762,
            bus0="632",bus1="671",
            length=0.61)
network.add("Line","6",
            x = 0.001,
            r = 0.191*0.762,
            bus0="671",bus1="684",
            length=0.0915)
network.add("Line","7",
            x = 0.001,
            r = 0.191*0.762,
            bus0="684",bus1="652",
            length=0.244)
network.add("Line","8",
            x = 0.001,
            r = 0.191*0.762,
            bus0="684",bus1="611",
            length=0.0915)
network.add("Line","9",
            x = 0.001,
            r = 0.191*0.762,
            bus0="671",bus1="692",
            length=0.0001)
network.add("Line","10",
            x = 0.001,
            r = 0.191*0.762,
            bus0="692",bus1="675",
            length=0.1525)
network.add("Line","11",
            x = 0.001,
            r = 0.191*0.762,
            bus0="671",bus1="680",
            length=0.305)

network.add("Generator","External Grid",bus="650",control="Slack")

# Multiply P and Q by 10^6
network.add("Load","1",bus="634", p_set=0.4, q_set=0.29)
network.add("Load","2",bus="645", p_set=0.170, q_set=0.125)
network.add("Load","3",bus="646", p_set=0.230, q_set=0.132)
network.add("Load","4",bus="652", p_set=0.128, q_set=0.086)
network.add("Load","5",bus="671", p_set=1.155, q_set=0.660)
network.add("Load","6",bus="675", p_set=0.843, q_set=0.462)
network.add("Load","7",bus="692", p_set=0.170, q_set=0.151)
network.add("Load","8",bus="611", p_set=0.170, q_set=0.080)

def radToDeg(dataframe):
    print("Voltage angles (deg):", '\n', (dataframe*180./np.pi).round(2))

def perUnitToVolts(dataframe):
#    network.buses_t.v_mag_pu['650'] = network.buses_t.v_mag_pu['650']*115/4.16
#    network.buses_t.v_mag_pu['634'] = network.buses_t.v_mag_pu['634']*0.48/4.16
    print("Voltage magnitudes (kV):", '\n', (dataframe).round(2))
    
    
def run_pf():
    network.pf(use_seed=True)
#    radToDeg(network.buses_t.v_ang)
    perUnitToVolts(network.buses_t.v_mag_pu)
#    print("Active power:", '\n', network.buses_t.p)
#    print("Reactive power:", '\n', network.buses_t.q)

run_pf()