#!/usr/bin/env bash
# Capture the Phase 1 network traffic (Modbus + MQTT) alongside the process
# telemetry. Phase 2 onwards derives the network-signal features S_net of
# PN-Fusion from these pcaps, so a benign capture must accompany the benign
# process baseline - they are two views of the same interval and are joined on
# wall-clock time.
#
#   sudo ./scripts/capture_traffic.sh 3600 data/raw/pcap
set -euo pipefail

DURATION="${1:-3600}"                  # seconds
OUTDIR="${2:-data/raw/pcap}"
IFACE="${IFACE:-any}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUTDIR}/benign_${STAMP}.pcap"

command -v tcpdump >/dev/null || { echo "install tcpdump: sudo apt install -y tcpdump"; exit 1; }
mkdir -p "${OUTDIR}"

echo "capturing ${DURATION}s of Modbus (502,5020) and MQTT (1883) on ${IFACE}"
echo "-> ${OUT}"
tcpdump -i "${IFACE}" -w "${OUT}" -s 0 \
        "tcp port 502 or tcp port 5020 or tcp port 1883" &
PID=$!
sleep "${DURATION}"
kill -INT "${PID}" 2>/dev/null || true
wait "${PID}" 2>/dev/null || true

sha256sum "${OUT}" | tee -a "${OUTDIR}/manifest.sha256"
echo "packets: $(tcpdump -r "${OUT}" 2>/dev/null | wc -l)"
