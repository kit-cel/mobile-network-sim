# Configuration

Configurations are validated YAML files. Start from [`radio.yaml`](../examples/radio.yaml) or [`radio_rrc.yaml`](../examples/radio_rrc.yaml).

| Field | Meaning |
| --- | --- |
| `seed` | Seed for reproducible radio episodes. |
| `time_resolution_ms` | Step duration: 1, 10, 20, or 40 ms. |
| `num_steps` | Steps in a standard run. |
| `dtypes` | Storage, internal, map, beam, and random-field precision. |
| `radio` | UE count, geometry, mobility, channel, antennas, and network. |
| `rrc` | UE RRC configuration; required for combined radio/RRC runs. |

A combined run requires `time_resolution_ms == rrc.t_res_ms`. Discrete RRC timers and update intervals must align to the RRC step duration.

The `RRCConfig` defaults for `n310` and `n311` are both **1**. The included `radio_rrc.yaml` sets both explicitly to **10** for its example scenario; an explicit YAML value overrides the default.

A3 handover settings live under `rrc.ho_event_config.a3`, including `ttt_ms`, `hys`, and `offset`. The standard handover mode requires positive simulated preparation and execution durations. `cho_idealized` permits zero trigger-time preparation by assuming prior preparation; it is not a full CHO implementation.
