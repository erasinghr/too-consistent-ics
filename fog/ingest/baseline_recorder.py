#!/usr/bin/env python3
"""Hybrid AI-Quantum Cognitive Security Framework Phase 1 - fog-tier baseline recorder.

Subscribes to the edge telemetry topic and writes every frame to time-rotated
CSV files under data/raw/baseline/. The resulting benign-operation dataset is
the primary artefact of Phase 1 and is consumed by:

    Phase 2 (RO1) - as the real-data reference distribution for WGAN-GP /
                    TabDDPM training and for the C2ST fidelity gate;
    Phase 3       - as the "normal" class for the physics-signal model of
                    PN-Fusion.

Each rotated file is accompanied by an entry in manifest.jsonl recording row
count, time span and SHA-256, so the dataset is auditable and DVC-trackable.

    python fog/ingest/baseline_recorder.py --config configs/twin.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

LOG = logging.getLogger("recorder")

FIELDS = ["seq", "ts", "ts_iso", "h1_m", "h2_m", "q_in_lpm", "q12_lpm",
          "q_out_lpm", "pump", "valve", "alarm", "src", "recv_ts"]
_RUN = True


def _stop(*_):
    global _RUN
    _RUN = False


class RotatingCsvWriter:
    def __init__(self, outdir: Path, rotate_minutes: int):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rotate_s = rotate_minutes * 60
        self.fh = None
        self.writer = None
        self.path = None
        self.opened_at = 0.0
        self.rows = 0
        self.t_first = None
        self.t_last = None

    def _open(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.outdir / f"baseline_{stamp}.csv"
        self.fh = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.opened_at = time.time()
        self.rows, self.t_first, self.t_last = 0, None, None
        LOG.info("recording to %s", self.path)

    def write(self, row: dict):
        if self.fh is None:
            self._open()
        self.writer.writerow(row)
        self.rows += 1
        self.t_first = self.t_first or row["ts"]
        self.t_last = row["ts"]
        self.fh.flush()
        if time.time() - self.opened_at >= self.rotate_s:
            self.close()

    def close(self):
        if self.fh is None:
            return
        self.fh.close()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        entry = {"file": self.path.name, "rows": self.rows, "sha256": digest,
                 "t_first": self.t_first, "t_last": self.t_last,
                 "closed_iso": datetime.now(timezone.utc).isoformat()}
        with open(self.outdir / "manifest.jsonl", "a") as m:
            m.write(json.dumps(entry) + "\n")
        LOG.info("closed %s (%d rows, sha256 %s...)", self.path.name, self.rows, digest[:12])
        self.fh = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twin.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = forever")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    cfg = yaml.safe_load(open(a.config))
    mq, st = cfg["mqtt"], cfg["storage"]
    outdir = Path(a.outdir or st["baseline_dir"])
    sink = RotatingCsvWriter(outdir, int(st["rotate_minutes"]))
    stats = {"n": 0, "bad": 0}

    import paho.mqtt.client as mqtt
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fog-recorder-1")
        v2 = True
    except AttributeError:
        client = mqtt.Client(client_id="fog-recorder-1")
        v2 = False

    def on_connect(c, u, flags, rc, props=None):
        LOG.info("connected to broker (rc=%s), subscribing to %s", rc, mq["topic"])
        c.subscribe(mq["topic"], qos=int(mq.get("qos", 0)))

    def on_message(c, u, msg):
        try:
            row = json.loads(msg.payload.decode())
            row["recv_ts"] = round(time.time(), 3)
            sink.write(row)
            stats["n"] += 1
            if stats["n"] % 300 == 0:
                LOG.info("%d frames recorded", stats["n"])
        except Exception as exc:  # malformed frame - count, never crash the recorder
            stats["bad"] += 1
            LOG.warning("dropped frame: %s", exc)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(mq["host"], int(mq["port"]), keepalive=30)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    client.loop_start()
    t_end = time.perf_counter() + a.duration if a.duration else float("inf")
    try:
        while _RUN and time.perf_counter() < t_end:
            time.sleep(0.2)
    finally:
        client.loop_stop()
        client.disconnect()
        sink.close()
        LOG.info("recorded %d frames (%d malformed)", stats["n"], stats["bad"])


if __name__ == "__main__":
    main()
