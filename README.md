# Too Consistent to Be Real — code for the two-sided residual test and the paired process–network dataset

This repository holds the code, configuration and calibration reports behind the paper

> A. Singh, V. Prakash, *Too Consistent to Be Real: A Two-Sided Residual Test for Fabricated Measurements in Industrial Control Systems, with a Paired Process–Network Dataset*, submitted to the International Journal of Critical Infrastructure Protection (2026).

The data (telemetry, packet captures, true-plant logs and the built plant model) are archived separately on Zenodo: **https://doi.org/10.5281/zenodo.24222734**.

It is an export of tag `paper1-ijcip-v1.0` (commit `ba0a1c1a78022cffa54e0a908fa1b955808dd501`) of the author's working repository, reduced to what the paper needs.

Release v1.1 adds the replay check reported in Section 8 of the paper (`scripts/phase3_attack_classes.py` and two report files, from commit `7db1eec5da5c9e55fbe8023bff26e831965eb7e6` of the same repository) and removes a duplicated `pathspec` pin from `requirements.txt`. Nothing else changed.

Release v1.2 adds the independent replication of the replay check and its splice-time diagnostic reported in Section 8 (`scripts/phase3_corpus.py`, `scripts/phase3_replay_seams.py` and two report files, from commit `44b337ea0313937417de78f67f5c57c28b48313e` of the same repository). Nothing else changed.

## What the testbed is

A two-tank gravity-drained level-control process with first-principles dynamics, written in Modelica and exported as an FMI 2.0 co-simulation unit, runs as a Modbus/TCP slave. An OpenPLC runtime executing an IEC 61131-3 Structured Text program closes the loop on a 100 ms scan. A separate publisher polls the twin at 1 Hz over its own Modbus connection and republishes over MQTT to a recorder, so a packet capture of the Modbus and MQTT ports is a second view of the same activity as the telemetry.

## Layout

| Path | Contents |
|---|---|
| `edge/twin/` | `TwoTank.mo` (plant model), `build_fmu.mos` (FMU build), `physics.py` (independent RK4 implementation), `twin_server.py` (co-simulation wrapper and Modbus slave) |
| `edge/plc/` | `two_tank.st` (control program), `soft_plc.py` (soft controller for validating the control law without the PLC runtime), `OPENPLC_SETUP.md` |
| `edge/publisher/` | `mqtt_publisher.py` (1 Hz Modbus poll, MQTT publication) |
| `fog/ingest/` | `baseline_recorder.py` (MQTT subscriber, rotated files with SHA-256 manifest) |
| `configs/twin.yaml` | single source of truth for plant parameters, set-points, Modbus map, timing and storage paths |
| `scripts/` | capture, dataset build and verification (Phase 1); residual floor, simulator, physics arms, attack ladder, network arm, live injector and scorers (Phase 2); replay check, its replication and splice-time diagnostic (Section 8) |
| `reports/` | JSON outputs behind the paper's tables (see below) |
| `docker-compose.yml`, `infra/` | broker and supporting services |
| `requirements*.txt`, `docs/versions_phase1.txt` | pinned Python dependencies and recorded tool versions |

## Environment

Tested on Ubuntu 22.04 under WSL2 with Docker Desktop, Python 3.11 (Conda environment), OpenModelica 1.27.0-dev, FMPy 0.3.21, DVC 3.51.2 and MLflow 2.14.3.
Three pins prevent silent breakage: `pyarrow==15.0.2` (MLflow 2.14.3 needs < 16), `setuptools<81` (later releases remove `pkg_resources`, which MLflow imports) and `pathspec<0.12` (1.1.1 breaks DVC 3.51.2).

**OpenPLC** is third-party software and is not included. The testbed used OpenPLC_v3 from https://github.com/thiagoralves/OpenPLC_v3.git at commit `b5d41356dab4aeadca0dd7ca64ba542f870b595d`; see `edge/plc/OPENPLC_SETUP.md`.

## Getting the data

Download the files from the Zenodo record and unzip them in the repository root. The paths inside each zip match this layout, so the files land where the scripts expect them:

```bash
unzip benign_telemetry.zip        # data/raw/baseline/, data/processed/
unzip benign_pcap.zip             # data/raw/pcap/
unzip live_attack_and_controls.zip  # data/raw/attack/, data/raw/pcap_attack/, data/raw/pcap_benign_ctrl/
unzip TwoTank_fmu.zip             # edge/twin/TwoTank.fmu
sha256sum -c SHA256SUMS.txt       # run in the folder holding the downloaded files
```

## Reproducing the results

| Step | Command | Produces |
|---|---|---|
| 1 | in `edge/twin/`: `omc build_fmu.mos` | the FMI 2.0 co-simulation unit |
| 2 | `sudo ./scripts/capture_traffic.sh 87000 data/raw/pcap`, then `timeout --signal=INT 86400 ./scripts/run_phase1.sh 0 1`, then **Start PLC** in the OpenPLC dashboard | 24 h benign telemetry and packet capture in real time |
| 3 | `python scripts/make_baseline_dataset.py --version v1`; `python scripts/verify_phase1.py --min-hours 24` | consolidated baseline and integrity report |
| 4 | `python scripts/crosscheck_fmu_rk4.py` | FMU versus independent RK4 trajectory comparison |
| 5 | `python scripts/phase2_floor_check.py` | residual-floor statistics |
| 6 | `python scripts/phase2_sim.py` | closed-loop simulator and its fidelity check |
| 7 | `python scripts/phase2_detector.py` | physics-arm calibration on 200 simulated days |
| 8 | `python scripts/phase2_attacks.py` | 350 simulated attack runs |
| 9 | `python scripts/phase2_pcap_inventory.py`; `python scripts/phase2_net_arm.py` | capture inventory and network-arm calibration |
| 10 | `./scripts/run_attack.sh` with `phase2_live_inject.py` and `phase2_truth_logger.py` | a live attack run: telemetry, packet capture and true-plant log |
| 11 | `python scripts/phase2_score_live.py`; `python scripts/phase2_net_arm.py --score`; `python scripts/phase2_net_arm_blocks.py` | live scores and the per-block attribution control |
| 12 | `python scripts/phase3_attack_classes.py`; `python scripts/phase3_attack_classes.py --classes P3-null --runs 100 --out reports/phase3_null_control.json` | replay check and its no-attack control, seed base 20261116 (Section 8) |
| 13 | `python scripts/phase3_corpus.py --build`; `python scripts/phase3_replay_seams.py` | independent replication of the replay check (160 runs, own seeds) and its splice-time diagnostic (Section 8); the build writes `data/processed/phase3_corpus_v1/` and also simulates attack families the paper does not report |

Each script documents its options (`--help`, or the header of the shell scripts). Three cautions:

- **Run the benign capture at a real-time factor of 1.** Modbus and MQTT inter-arrival timings are part of the network features.
- **Do not pass a duration to `run_phase1.sh`.** A bounded simulation loop stops the plant but not the Modbus server, which keeps serving its last values while the recorder appends flat-line rows. Run unbounded and stop it externally, as above.
- **Start the PLC program by hand.** The launcher does not; without it the plant drains open-loop.

## Report files

| File | Supports |
|---|---|
| `reports/phase1_verification.json` | integrity of the benign capture |
| `reports/phase2_detector_calibration.json` | physics-arm thresholds (Arm A, one-sided; Arm B, two-sided) |
| `reports/phase2_sim_fidelity.json` | simulator against the released day |
| `reports/phase2_attack_runs.json` | the 350 simulated attack runs |
| `reports/phase2_pcap_inventory.json` | connections, function codes and timing of the benign capture |
| `reports/phase2_net_arm_calibration.json` | network-arm thresholds |
| `reports/phase2_net_arm_score_*.json`, `reports/phase2_net_arm_blocks.json` | network-arm scores on the live run and the attribution controls |
| `reports/phase3_attack_classes.json` | the replay check of Section 8 (the replay classes; freeze and scale runs from the same script are not reported in the paper) |
| `reports/phase3_null_control.json` | its no-attack control under the same protocol |
| `reports/phase3_corpus_v1_summary.json` | the replication sample: per-rung Arm A and Arm B results of the corpus build (Section 8 reports the replay rows) |
| `reports/phase3_replay_seams.json` | the splice-time diagnostic of the replicated replay runs |
| `reports/phase2_physics_arms_calibration.json` | an earlier calibration step, superseded by `phase2_detector_calibration.json` (as is `scripts/phase2_calibrate_physics_arms.py`) |

## Licence and citation

Code: MIT licence (`LICENSE`). Data on Zenodo: CC BY 4.0. Please cite the paper and the Zenodo record; `CITATION.cff` holds the details.
