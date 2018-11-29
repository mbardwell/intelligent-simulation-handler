## Example of transformer with non-trivial phase shift and tap ratio
#
#This example is a copy of pandapower's minimal example.
import logging
logging.basicConfig(level=logging.ERROR)

import pypsa
pypsa.pf.logger.setLevel(logging.ERROR)
import numpy as np

network = pypsa.Network()

network.add("Bus","MV1 bus",v_nom=12.47,v_mag_pu_set=1.00)
network.add("Bus","MV2 bus",v_nom=12.47,v_mag_pu_set=1.00)
network.add("Bus","LV1 bus",v_nom=24.9)
network.add("Bus","LV2 bus",v_nom=24.9)

#network.add("Transformer","MV-LV trafo",type="25 MVA 110/10 kV",bus0="MV2 bus",bus1="LV1 bus")
network.add("Transformer","TF name",
            x = 0.06,
            r = 0.01,
            s_nom = 6, # in MVA
            s_max_pu = 1,
            bus0="MV2 bus",
            bus1="LV1 bus"
            )

# 2000 ft = 0.61 km, 2500 ft = 0.762 km
# since we are calculating delta-delta, use phase cable impedance
# 0.306 ohm/mile = 0.191 ohm/km, 0.721 inch = 18.3 mm -> 263 mm^2 cross sect
network.add("Line","MV cable",
            x = 0.001,
            r = 0.191*0.61,
            bus0="MV1 bus",bus1="MV2 bus",
            length=0.61)
network.add("Line","LV cable",
            x = 0.001,
            r = 0.191*0.762,
            bus0="LV1 bus",bus1="LV2 bus",length=0.762)

network.add("Generator","External Grid",bus="MV1 bus",control="Slack")

# P = 1.8 MW * 3 = 5.4 MW 
# S = P/cos(theta) = 5.4 MW/cos(0.9) = 8.687 MVA
# Q = sqrt(S^2 - P^2) = sqrt(8.7^2 - 5.4 ^2) = 6.8 MVAR lagging (inductive, so +)
network.add("Load","LV load",bus="LV2 bus", p_set=5.4, q_set=6.8)

def run_pf():
    network.lpf()
    network.pf(use_seed=True)
    print("Voltage angles:", network.buses_t.v_ang*180./np.pi)
    print("Voltage magnitudes:", network.buses_t.v_mag_pu)
    print("Active power:", network.buses_t.p)
    print("Reactive power:", network.buses_t.q)
    # I = P(in W)/(V*PF*sqrt(3)) = 5.4e6/(0.63*4.16e3*0.9*sqrt(3)) = 1.321 kA

run_pf()