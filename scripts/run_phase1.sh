#!/usr/bin/env bash
# Bring up the full Phase 1 edge/fog stack in one command and tear it down
# cleanly on Ctrl-C. Assumes: conda env `cps` active, docker services running
# (mosquitto at least), OpenPLC configured per edge/plc/OPENPLC_SETUP.md.
#
#   ./scripts/run_phase1.sh                 # real time, runs until Ctrl-C
#   ./scripts/run_phase1.sh 3600 1          # 1 h of process time at 1x
#   ./scripts/run_phase1.sh 86400 60        # 24 h of process time at 60x
set -euo pipefail

DURATION="${1:-0}"          # process seconds; 0 = unbounded
RTF="${2:-1}"               # realtime factor
CFG="${CFG:-configs/twin.yaml}"
ENGINE="${ENGINE:-fmu}"
nc -z localhost 1883 || { echo "ERROR: MQTT broker unreachable on 1883 - start docker compose first"; exit 1; }

ss -ltn | grep -q ":5020 " && { echo "ERROR: port 5020 already in use - another twin is running"; exit 1; }

mkdir -p logs data/raw/baseline

cleanup() {
  echo; echo "stopping Phase 1 stack..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "stopped. logs in ./logs, data in data/raw/baseline"
}
trap cleanup EXIT INT TERM
PIDS=()

echo "[1/3] digital twin (engine=${ENGINE}, rtf=${RTF}, duration=${DURATION})"
python edge/twin/twin_server.py --config "$CFG" --engine "$ENGINE" \
       --realtime-factor "$RTF" --duration "$DURATION" > logs/twin.log 2>&1 &
PIDS+=($!); sleep 3

echo "[2/3] fog baseline recorder"
python fog/ingest/baseline_recorder.py --config "$CFG" > logs/recorder.log 2>&1 &
PIDS+=($!); sleep 1

echo "[3/3] edge MQTT publisher"
python edge/publisher/mqtt_publisher.py --config "$CFG" > logs/publisher.log 2>&1 &
PIDS+=($!)

echo
echo "stack up. OpenPLC must be started separately (Dashboard -> Start PLC)."
echo "follow with:  tail -f logs/twin.log logs/publisher.log logs/recorder.log"
wait
