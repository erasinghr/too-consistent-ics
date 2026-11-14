#!/usr/bin/env bash
# Phase 2.4 - bring up the stack with the live Modbus injector in the control path.
#
# Wiring: the twin is moved to :5030 and the injector proxy takes :5020, so OpenPLC and
# the publisher reach the twin THROUGH the proxy without any change to their own config.
#   OpenPLC / publisher  ->  :5020 (proxy)  ->  :5030 (twin)
# The normal stack (configs/twin.yaml, run_phase1.sh) is left untouched; this uses its own
# generated configs. Revert by just running run_phase1.sh again.
#
#   ./scripts/run_attack.sh <rung> [arm_s] [duration_s]
#   ./scripts/run_attack.sh R0-step 999999 0     # proxied PASSTHROUGH smoke test (never arms)
#   ./scripts/run_attack.sh R3-fab-matched 120 600
#
# Start the pcap capture in another terminal BEFORE this (see the runbook). OpenPLC must
# already be scanning; it reconnects to :5020 (now the proxy) on its own.
set -euo pipefail

RUNG="${1:?usage: run_attack.sh <rung> [arm_s] [duration_s]}"
ARM="${2:-120}"
DUR="${3:-600}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNID="${RUNG}_${STAMP}"
OUTDIR="data/raw/attack/${RUNID}"
TWINCFG="configs/_attack_twin.yaml"
PUBCFG="configs/_attack_${RUNID}.yaml"

command -v python >/dev/null || { echo "ERROR: activate the cps env first (conda activate cps)"; exit 1; }
nc -z localhost 1883 || { echo "ERROR: MQTT broker unreachable on 1883 - start docker compose first"; exit 1; }
ss -ltn | grep -q ":5020 " && { echo "ERROR: :5020 in use - stop the normal stack (Ctrl-C run_phase1.sh) first"; exit 1; }
ss -ltn | grep -q ":5030 " && { echo "ERROR: :5030 in use - another twin is already there"; exit 1; }
[ -f scripts/phase2_live_inject.py ] || { echo "ERROR: scripts/phase2_live_inject.py missing"; exit 1; }

mkdir -p logs "$OUTDIR"
# twin config: same as production but bound to 5030
sed 's/port: 5020/port: 5030/' configs/twin.yaml > "$TWINCFG"
# publisher/recorder config: production modbus port (5020 -> proxy), attack output dir
sed "s#baseline_dir: data/raw/baseline#baseline_dir: ${OUTDIR}#" configs/twin.yaml > "$PUBCFG"
# sanity: the twin config's modbus port must now be 5030 (matches "port: 5030", any spacing)
grep -Eq '^[[:space:]]*port:[[:space:]]*5030([[:space:]]|$)' "$TWINCFG" \
  || { echo "ERROR: could not set twin port to 5030 in $TWINCFG"; exit 1; }

PIDS=()
cleanup() {
  echo; echo "stopping attack stack..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  rm -f "$PUBCFG"
  echo "stopped. telemetry in ${OUTDIR}, injector sidecar in reports/, logs in ./logs"
}
trap cleanup EXIT INT TERM

echo "[1/5] twin on :5030  (log: logs/attack_twin.log)"
python edge/twin/twin_server.py --config "$TWINCFG" --engine fmu --realtime-factor 1 --duration 0 \
       > logs/attack_twin.log 2>&1 &
PIDS+=($!); sleep 3
ss -ltn | grep -q ":5030 " || { echo "ERROR: twin did not come up on 5030 - see logs/attack_twin.log"; exit 1; }

echo "[2/5] ground-truth logger (polls the TRUE twin on :5030 directly) -> ${OUTDIR}/truth.csv"
python scripts/phase2_truth_logger.py --port 5030 --out "${OUTDIR}/truth.csv" > logs/attack_truth.log 2>&1 &
PIDS+=($!); sleep 1

echo "[3/5] injector  :5020 -> :5030  (rung=${RUNG} arm=${ARM}s dur=${DUR}s; log: logs/attack_proxy.log)"
python scripts/phase2_live_inject.py --rung "$RUNG" --arm "$ARM" --duration "$DUR" \
       --listen-port 5020 --twin-port 5030 > logs/attack_proxy.log 2>&1 &
PIDS+=($!); sleep 2
ss -ltn | grep -q ":5020 " || { echo "ERROR: proxy did not bind 5020 - see logs/attack_proxy.log"; exit 1; }

echo "[4/5] fog baseline recorder -> ${OUTDIR}  (log: logs/attack_recorder.log)"
python fog/ingest/baseline_recorder.py --config "$PUBCFG" > logs/attack_recorder.log 2>&1 &
PIDS+=($!); sleep 1

echo "[5/5] edge MQTT publisher (polls :5020 = proxy)  (log: logs/attack_publisher.log)"
python edge/publisher/mqtt_publisher.py --config "$PUBCFG" > logs/attack_publisher.log 2>&1 &
PIDS+=($!)

echo
echo "attack stack up for run ${RUNID}."
echo "  onset and events:  tail -f logs/attack_proxy.log"
echo "  TRUE plant state:  tail -f ${OUTDIR}/truth.csv   (real levels while telemetry may be fabricated)"
echo "  plant + control:   tail -f ${OUTDIR}/baseline_*.csv"
echo "OpenPLC reconnects to :5020 (the proxy) by itself. Ctrl-C here stops everything."
wait
