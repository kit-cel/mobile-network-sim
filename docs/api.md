# Python API

| Import | Purpose |
| --- | --- |
| `mobile_network_sim.radio.SimulationConfig` | Validate radio-only configuration. |
| `mobile_network_sim.radio.RadioEnvironment` | Advance radio state one step at a time. |
| `mobile_network_sim.radio.RadioSimulationRunner` | Run and optionally save radio episodes. |
| `mobile_network_sim.radio.RadioStepResult` | Radio measurements for one step. |
| `mobile_network_sim.rrc.RRCConfig` | Validate RRC timers and handover settings. |
| `mobile_network_sim.rrc.RRCStepUE` | Advance one UE using RSRP/SINR samples. |
| `mobile_network_sim.runners.RadioRRCSimulationConfig` | Validate a combined configuration. |
| `mobile_network_sim.runners.RadioRRCSimulationRunner` | Advance radio and all RRC UEs together. |
| `mobile_network_sim.runners.RadioRRCStepResult` | Radio and RRC results for one step. |

The combined runner supports `reset(seed=...)`, `step()`, `run(num_steps=...)`, and `close()`. With external control, pass one action per UE to `step(actions=...)`. `RRCStepUE` exposes `step(...)`, `snapshot()`, and `get_flags()`. See the class docstrings for all parameters and result fields.
