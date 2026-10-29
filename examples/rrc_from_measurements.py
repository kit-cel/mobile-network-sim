"""Drive RRC independently from stored or generated measurements."""

import numpy as np

from mobile_network_sim.runners import RadioRRCSimulationConfig
from mobile_network_sim.rrc import RRCStepUE

config = RadioRRCSimulationConfig.from_yaml("examples/radio_rrc.yaml").rrc
ue = RRCStepUE(ue_id=0, rrc_config=config)
ue.step(
    time_ms=0,
    rsrp_dbm=np.asarray([-80.0, -90.0]),
    sinr_db=np.asarray([10.0, 2.0]),
)
print(ue.snapshot())
