#!/usr/bin/env python
"""Phase 2, step 2.3a - inventory of the benign packet capture (standard library only).

Before choosing features for the network-only arm, establish what the capture
actually contains:
  * file: link type, packets, time span, capture gaps
  * capture layer: interfaces / packet directions (a capture on `any` can see a
    loopback or bridged packet more than once), and how many packets are exact
    duplicates of one seen within 50 ms - duplicates are counted once everywhere else
  * conversations: endpoints, packets each way, bytes, first/last seen
  * Modbus/TCP (ports 502, 5020), per client connection: function codes, exception
    responses, address ranges, unit ids; per function code the request inter-arrival,
    the request -> response latency (matched on transaction id) and requests per hour
  * MQTT (1883): packet types per conversation (all control packets in a segment)
  * TCP lifecycle: SYN, FIN and RST counts, new connections per hour
No third-party packages: classic pcap (not pcapng), IPv4 and IPv6, TCP and UDP.

    python scripts/phase2_pcap_inventory.py data/raw/pcap/benign_20260902T005748Z.pcap
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODBUS_PORTS = {502, 5020}
MQTT_PORTS = {1883}
MQTT_TYPES = {1: "CONNECT", 2: "CONNACK", 3: "PUBLISH", 4: "PUBACK", 5: "PUBREC", 6: "PUBREL", 7: "PUBCOMP",
              8: "SUBSCRIBE", 9: "SUBACK", 10: "UNSUBSCRIBE", 11: "UNSUBACK", 12: "PINGREQ", 13: "PINGRESP",
              14: "DISCONNECT", 15: "AUTH"}
MODBUS_FC = {1: "read coils", 2: "read discrete inputs", 3: "read holding registers", 4: "read input registers",
             5: "write single coil", 6: "write single register", 15: "write multiple coils",
             16: "write multiple registers", 23: "read/write multiple registers", 43: "device identification"}
SLL_PKTTYPE = {0: "to-us", 1: "broadcast", 2: "multicast", 3: "other-host", 4: "outgoing"}


# ---------------------------------------------------------------- pcap reading
def read_pcap(path: Path):
    """Yield (ts, linktype, frame bytes). Classic pcap, either byte order, us or ns."""
    with open(path, "rb") as f:
        gh = f.read(24)
        magic = gh[:4]
        if magic == b"\x0a\x0d\x0d\x0a":
            raise SystemExit("this is pcapng - convert first: tcpdump -r in.pcapng -w out.pcap")
        table = {b"\xd4\xc3\xb2\xa1": ("<", 1e-6), b"\xa1\xb2\xc3\xd4": (">", 1e-6),
                 b"\x4d\x3c\xb2\xa1": ("<", 1e-9), b"\xa1\xb2\x3c\x4d": (">", 1e-9)}
        if magic not in table:
            raise SystemExit(f"not a pcap file (magic {magic.hex()})")
        end, frac = table[magic]
        _, _, _, _, snaplen, linktype = struct.unpack(end + "HHiIII", gh[4:])
        rec = struct.Struct(end + "IIII")
        yield None, linktype, snaplen                 # header info first
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            sec, sub, incl, _orig = rec.unpack(h)
            yield sec + sub * frac, linktype, f.read(incl)


def link_payload(linktype: int, fr: bytes):
    """Return (ethertype, l3 bytes, capture tag) or None. The tag records interface/direction."""
    if linktype == 1:                                  # Ethernet
        if len(fr) < 14:
            return None
        et, off = struct.unpack("!H", fr[12:14])[0], 14
        while et in (0x8100, 0x88A8) and len(fr) >= off + 4:   # VLAN tags
            et, off = struct.unpack("!H", fr[off + 2:off + 4])[0], off + 4
        return et, fr[off:], "eth"
    if linktype == 113:                                # Linux cooked v1 (tcpdump -i any)
        if len(fr) < 16:
            return None
        ptype = struct.unpack("!H", fr[0:2])[0]
        return struct.unpack("!H", fr[14:16])[0], fr[16:], f"sll:{SLL_PKTTYPE.get(ptype, ptype)}"
    if linktype == 276:                                # Linux cooked v2
        if len(fr) < 20:
            return None
        proto, _, ifindex = struct.unpack("!HHI", fr[0:8])
        ptype = fr[10]
        return proto, fr[20:], f"sll2:if{ifindex}:{SLL_PKTTYPE.get(ptype, ptype)}"
    if linktype == 0:                                  # BSD loopback
        fam = struct.unpack("<I", fr[0:4])[0] if len(fr) >= 4 else -1
        return (0x0800 if fam == 2 else 0x86DD), fr[4:], "null"
    if linktype in (12, 14, 101, 228, 229):            # raw IP
        if not fr:
            return None
        v = fr[0] >> 4
        return (0x0800 if v == 4 else 0x86DD), fr, "raw"
    return None


def parse_ip(et: int, b: bytes):
    """Return (src, dst, proto, l4 bytes, ip_id) or None."""
    if et == 0x0800 and len(b) >= 20:
        ihl = (b[0] & 0x0F) * 4
        tot = struct.unpack("!H", b[2:4])[0]
        ip_id = struct.unpack("!H", b[4:6])[0]
        proto = b[9]
        src = ".".join(map(str, b[12:16]))
        dst = ".".join(map(str, b[16:20]))
        return src, dst, proto, b[ihl:tot if tot >= ihl else None], ip_id
    if et == 0x86DD and len(b) >= 40:
        proto = b[6]
        src = ":".join(b[8:24].hex()[i:i + 4] for i in range(0, 32, 4))
        dst = ":".join(b[24:40].hex()[i:i + 4] for i in range(0, 32, 4))
        return src, dst, proto, b[40:], None
    return None


# ---------------------------------------------------------------- statistics helpers
class Gaps:
    """Streaming inter-arrival summary with a coarse histogram (log-spaced bins)."""
    EDGES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.095, 0.105, 0.12, 0.2, 0.5, 0.9, 0.99, 1.01, 1.1, 2, 5, 30, 1e9]

    def __init__(self):
        self.last = None
        self.n = 0
        self.s = self.s2 = 0.0
        self.mx = 0.0
        self.hist = [0] * len(self.EDGES)

    def add(self, t):
        if self.last is not None:
            g = t - self.last
            self.n += 1
            self.s += g
            self.s2 += g * g
            self.mx = max(self.mx, g)
            for i, e in enumerate(self.EDGES):
                if g < e:
                    self.hist[i] += 1
                    break
        self.last = t

    def summary(self):
        if self.n == 0:
            return None
        mu = self.s / self.n
        sd = math.sqrt(max(self.s2 / self.n - mu * mu, 0.0))
        lo = [0.0] + self.EDGES[:-1]
        return {"n": self.n, "mean_s": round(mu, 5), "sd_s": round(sd, 5), "max_s": round(self.mx, 3),
                "hist": {f"{lo[i]:g}-{self.EDGES[i]:g}": c for i, c in enumerate(self.hist) if c}}


class Lat(Gaps):
    """Request -> response latency summary (log-spaced bins, seconds)."""
    EDGES = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1e9]

    def add_value(self, g):
        self.n += 1
        self.s += g
        self.s2 += g * g
        self.mx = max(self.mx, g)
        for i, e in enumerate(self.EDGES):
            if g < e:
                self.hist[i] += 1
                break


def mqtt_packets(payload: bytes):
    """Yield the control-packet types in one TCP segment (several can share a segment)."""
    off = 0
    while off < len(payload):
        mtype = payload[off] >> 4
        mult, rem, i = 1, 0, off + 1
        while i < len(payload):
            byte = payload[i]
            rem += (byte & 0x7F) * mult
            mult *= 128
            i += 1
            if not byte & 0x80 or mult > 128 ** 3:
                break
        yield mtype
        nxt = i + rem
        if nxt <= off:
            break
        off = nxt


# ---------------------------------------------------------------- main pass
def inventory(path: Path, dup_window: float = 0.05) -> dict:
    gen = read_pcap(path)
    _, linktype, snaplen = next(gen)
    n = 0
    t_first = t_last = None
    big_gaps = []
    tags = Counter()
    ethertypes = Counter()
    protos = Counter()
    conv = {}
    recent = {}                       # duplicate detector: key -> last time seen
    dups = 0
    modbus = defaultdict(Counter)     # conv -> Counter of (dir, fc)
    modbus_units = defaultdict(Counter)
    modbus_ranges = defaultdict(Counter)
    modbus_req_gaps = defaultdict(Gaps)    # (conn, fc) -> inter-arrival of requests
    modbus_lat = defaultdict(Lat)          # (conn, fc) -> request -> response latency
    modbus_hourly = defaultdict(Counter)   # (conn, fc) -> hour -> requests
    pending = {}                           # (conn, transaction id) -> request time
    unmatched = Counter()
    modbus_exc = Counter()
    modbus_partial = Counter()
    mqtt = defaultdict(Counter)
    tcp_flags = Counter()
    syn_by_hour = Counter()
    t_prev = None

    for ts, lt, fr in gen:
        n += 1
        if n % 1_000_000 == 0:
            print(f"    {n / 1e6:.0f} M packets ...", flush=True)
        if t_first is None:
            t_first = ts
        if t_prev is not None and ts - t_prev > 5.0:
            big_gaps.append((round(t_prev, 3), round(ts - t_prev, 3)))
        t_prev = t_last = ts
        lp = link_payload(lt, fr)
        if lp is None:
            tags["unparsed-link"] += 1
            continue
        et, l3, tag = lp
        tags[tag] += 1
        ethertypes[hex(et)] += 1
        ip = parse_ip(et, l3)
        if ip is None:
            continue
        src, dst, proto, l4, ip_id = ip
        protos[{6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}.get(proto, str(proto))] += 1
        sport = dport = None
        payload = b""
        flags = 0
        seq = None
        if proto == 6 and len(l4) >= 20:
            sport, dport, seq, _ack, offr, flags = struct.unpack("!HHIIBB", l4[:14])
            payload = l4[(offr >> 4) * 4:]
        elif proto == 17 and len(l4) >= 8:
            sport, dport = struct.unpack("!HH", l4[:4])
            payload = l4[8:]

        # duplicates: the same IP packet seen again within dup_window (another interface/direction)
        key = (src, dst, proto, ip_id, sport, dport, seq, len(payload))
        if ip_id is not None:
            prev = recent.get(key)
            recent[key] = ts
            if len(recent) > 200_000:
                recent = {k: v for k, v in recent.items() if ts - v <= dup_window}
            if prev is not None and ts - prev <= dup_window:
                dups += 1
                continue          # count each IP packet once in everything below

        a, b = (src, sport), (dst, dport)
        ck = (proto, *(sorted([a, b], key=lambda x: (str(x[0]), x[1] or 0))))
        c = conv.get(ck)
        if c is None:
            c = conv[ck] = {"pk_ab": 0, "pk_ba": 0, "bytes": 0, "first": ts, "last": ts}
        c["pk_ab" if (src, sport) == ck[1] else "pk_ba"] += 1
        c["bytes"] += len(l3)
        c["last"] = ts

        if proto == 6:
            if flags & 0x02 and not flags & 0x10:
                tcp_flags["SYN"] += 1
                syn_by_hour[int((ts - t_first) // 3600)] += 1
            if flags & 0x01:
                tcp_flags["FIN"] += 1
            if flags & 0x04:
                tcp_flags["RST"] += 1
        if not payload:
            continue

        name = f"{src}:{sport} -> {dst}:{dport}"
        if proto == 6 and (sport in MODBUS_PORTS or dport in MODBUS_PORTS):
            is_req = dport in MODBUS_PORTS
            server = f"{dst}:{dport}" if is_req else f"{src}:{sport}"
            client = f"{src}:{sport}" if is_req else f"{dst}:{dport}"
            cname = f"{client} -> {server}"
            off = 0
            while off + 8 <= len(payload):
                tid, pid, length, unit = struct.unpack("!HHHB", payload[off:off + 7])
                if pid != 0 or length < 2:
                    modbus_partial[cname] += 1
                    break
                if off + 6 + length > len(payload):
                    modbus_partial[cname] += 1
                    break
                fc = payload[off + 7]
                d = "req" if is_req else "rsp"
                modbus[cname][(d, fc)] += 1
                modbus_units[cname][unit] += 1
                if fc >= 0x80:
                    modbus_exc[(cname, fc & 0x7F, payload[off + 8] if off + 8 < len(payload) else None)] += 1
                if is_req and fc in (1, 2, 3, 4, 5, 6, 15, 16) and off + 12 <= len(payload):
                    addr, qty = struct.unpack("!HH", payload[off + 8:off + 12])
                    if fc in (5, 6):
                        modbus_ranges[cname][(fc, addr, 1)] += 1
                    else:
                        modbus_ranges[cname][(fc, addr, qty)] += 1
                if is_req:
                    modbus_req_gaps[(cname, fc)].add(ts)
                    modbus_hourly[(cname, fc)][int((ts - t_first) // 3600)] += 1
                    pending[(cname, tid)] = (ts, fc)
                    if len(pending) > 100_000:          # responses that never came: drop the oldest
                        pending = {k: v for k, v in pending.items() if ts - v[0] < 10.0}
                else:
                    req = pending.pop((cname, tid), None)
                    if req is None:
                        unmatched[cname] += 1
                    else:
                        modbus_lat[(cname, req[1])].add_value(ts - req[0])
                off += 6 + length
        elif proto == 6 and (sport in MQTT_PORTS or dport in MQTT_PORTS):
            for mtype in mqtt_packets(payload):
                mqtt[name][MQTT_TYPES.get(mtype, str(mtype))] += 1

    span = (t_last - t_first) if n else 0.0
    convs = sorted(conv.items(), key=lambda kv: -(kv[1]["pk_ab"] + kv[1]["pk_ba"]))
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")
    return {
        "file": str(path), "bytes": path.stat().st_size, "linktype": linktype, "snaplen": snaplen,
        "packets": n, "first": iso(t_first) if n else None, "last": iso(t_last) if n else None,
        "span_s": round(span, 3), "gaps_over_5s": big_gaps[:50], "n_gaps_over_5s": len(big_gaps),
        "capture_tags": dict(tags.most_common()), "ethertypes": dict(ethertypes), "ip_protocols": dict(protos),
        "duplicates_within_50ms": dups,
        "conversations": [{
            "proto": {6: "TCP", 17: "UDP"}.get(k[0], k[0]),
            "a": f"{k[1][0]}:{k[1][1]}", "b": f"{k[2][0]}:{k[2][1]}",
            "packets_a_to_b": v["pk_ab"], "packets_b_to_a": v["pk_ba"], "bytes": v["bytes"],
            "first": iso(v["first"]), "last": iso(v["last"]),
            "rate_per_s": (round((v["pk_ab"] + v["pk_ba"]) / (v["last"] - v["first"]), 3)
                           if v["last"] - v["first"] >= 1.0 else None)}
            for k, v in convs],
        "tcp": {"flags": dict(tcp_flags), "syn_per_hour": {str(h): c for h, c in sorted(syn_by_hour.items())}},
        "modbus": {c: {
            "function_codes": {f"{d} {fc} ({MODBUS_FC.get(fc & 0x7F, '?')}{', EXCEPTION' if fc >= 0x80 else ''})": k
                               for (d, fc), k in sorted(modbus[c].items())},
            "unit_ids": dict(modbus_units[c]),
            "request_ranges (fc, start, qty)": {str(k): v for k, v in modbus_ranges[c].most_common(20)},
            "per_function_code": {str(fc): {
                "request_interarrival": modbus_req_gaps[(c, fc)].summary(),
                "latency": modbus_lat[(c, fc)].summary(),
                "requests_per_hour_min_max": [min(modbus_hourly[(c, fc)].values()),
                                              max(modbus_hourly[(c, fc)].values())] if modbus_hourly[(c, fc)] else None,
                "requests_per_hour": {str(h): k for h, k in sorted(modbus_hourly[(c, fc)].items())},
            } for (cc, fc) in sorted(modbus_req_gaps) if cc == c},
            "unmatched_responses": unmatched[c],
            "partial_frames": modbus_partial[c]} for c in modbus},
        "modbus_exceptions": {f"{c} fc{fc} code{code}": k for (c, fc, code), k in modbus_exc.items()},
        "mqtt": {c: dict(v) for c, v in mqtt.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("--out", default="reports/phase2_pcap_inventory.json")
    ap.add_argument("--top", type=int, default=25, help="conversations to print")
    a = ap.parse_args()
    path = Path(a.pcap)
    t0 = datetime.now()
    print(f"reading {path} ({path.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    rep = inventory(path)
    rep["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"    done in {(datetime.now() - t0).total_seconds():.0f} s\n")

    print(f"link type {rep['linktype']}, {rep['packets']:,} packets, {rep['first']} -> {rep['last']} "
          f"({rep['span_s'] / 3600:.3f} h), gaps > 5 s: {rep['n_gaps_over_5s']}")
    print(f"capture tags: {rep['capture_tags']}")
    print(f"IP protocols: {rep['ip_protocols']}   exact duplicates within 50 ms: {rep['duplicates_within_50ms']:,} "
          f"(counted once in everything below)")
    print(f"TCP flags: {rep['tcp']['flags']}")
    print(f"SYN per hour: {rep['tcp']['syn_per_hour']}")

    print(f"\ntop {a.top} conversations by packets")
    print(f"  {'proto':<6}{'endpoint a':<26}{'endpoint b':<26}{'a->b':>10}{'b->a':>10}{'pkt/s':>9}   first -> last")
    for c in rep["conversations"][:a.top]:
        rate = "-" if c["rate_per_s"] is None else f"{c['rate_per_s']:.2f}"
        print(f"  {c['proto']:<6}{c['a']:<26}{c['b']:<26}{c['packets_a_to_b']:>10}{c['packets_b_to_a']:>10}"
              f"{rate:>9}   {c['first'][11:19]} -> {c['last'][11:19]}")
    print(f"  ({len(rep['conversations'])} conversations in total)")

    for c, m in rep["modbus"].items():
        print(f"\nModbus {c}")
        for k, v in m["function_codes"].items():
            print(f"    {k:<52}{v:>10,}")
        print(f"    unit ids {m['unit_ids']}   partial frames {m['partial_frames']}   "
              f"unmatched responses {m['unmatched_responses']}")
        print(f"    request ranges {m['request_ranges (fc, start, qty)']}")
        for fc, d in m["per_function_code"].items():
            g, l, hr = d["request_interarrival"], d["latency"], d["requests_per_hour_min_max"]
            print(f"    fc {fc}: requests per hour {hr[0]:,}-{hr[1]:,} (first and last hour are partial)")
            if g:
                print(f"        inter-arrival mean {g['mean_s']} s, sd {g['sd_s']} s, max {g['max_s']} s")
                print(f"            histogram (s): {g['hist']}")
            if l:
                print(f"        response latency mean {l['mean_s'] * 1000:.3f} ms, sd {l['sd_s'] * 1000:.3f} ms, "
                      f"max {l['max_s'] * 1000:.1f} ms")
                print(f"            histogram (s): {l['hist']}")
    if rep["modbus_exceptions"]:
        print(f"\nModbus exceptions: {rep['modbus_exceptions']}")
    for c, m in rep["mqtt"].items():
        print(f"\nMQTT {c}: {m}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
