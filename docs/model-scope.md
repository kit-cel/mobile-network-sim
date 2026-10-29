# Model scope

`mobile-network-sim` generates step-based radio measurements and can advance a simplified UE RRC model on the same time grid. It supports radio-only runs, combined radio/RRC runs, and RRC runs driven by stored measurements.

## Radio model

- Configurable base-station layouts, UE mobility, antennas, and beamforming.
- Large-scale UMa and RMa channel models with configurable line of sight and shadowing.
- Per-step geometry and link measurements, including RSRP and SINR. Radio-only runs can save results to HDF5.

## RRC model

- One state machine per UE, with initial access, A3-triggered handover, radio-link monitoring, handover failure, and simulated RLF recovery.
- Configurable measurement filtering, timers, and handover preparation and execution times.
- Autonomous handover or externally supplied actions (`binary_report` and `target_cell`).

`cho_idealized` assumes that target preparation has already happened; it does not model candidate admission or preparation signaling. The RRC model does not implement full signaling, bearer setup, or network-side protocol entities. See [RRC model](rrc.md) for its states and behavior.

## Time and output

Radio and RRC advance in discrete steps. A combined run requires matching step durations; configured timers and update intervals must align with the RRC step duration. The CLI saves radio-only samples to HDF5 and a combined run's final RRC summary to JSON. Per-step combined results are available through the Python runner. See [Usage](usage.md) and [Configuration](configuration.md).
