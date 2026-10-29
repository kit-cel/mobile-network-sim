# RRC model

The simulator implements one step-based UE RRC state machine per UE. It models initial access, A3-triggered handover, N310/N311 and T310 radio-link monitoring, handover failure, and simulated RLF recovery. It is not a complete NR RRC protocol stack.

In a combined run, radio produces the current RSRP and SINR vectors before each UE's RRC state advances. `RRCStepUE.step(...)` can also consume stored measurements directly. Its `snapshot()` returns the current state, serving cell, timers, and event flags.

Autonomous operation evaluates filtered RSRP against the configured A3 offset, hysteresis, and time-to-trigger. Policy-controlled operation disables autonomous handover triggering and accepts an action through the runner's `action_mode` (`binary_report` or `target_cell`). Radio-link monitoring remains active.

The states are `IDLE`, `CONNECTED_NORMAL`, `CONNECTED_HO_PREP`, `CONNECTED_HO_EXEC`, `CONNECTED_PHY_PROBLEM`, and `RLF_RECOVERY`. The user plane remains available during preparation and is interrupted during execution and failure handling. The model does not simulate full signaling, bearer setup, or network-side protocol entities.
