# OpenPLC configuration for Phase 1

The digital twin is a Modbus **slave**; OpenPLC polls it as a *slave device*.
This mirrors a real deployment, where the PLC polls remote I/O, and it is the
reason the ST program addresses I/O from `%IW100` / `%QX100.0` upwards —
OpenPLC allocates slave-device I/O starting at index 100.

## 1. Start OpenPLC

```bash
docker compose -f infra/docker-compose.phase1.yml up -d openplc
# web UI: http://localhost:8080   default login: openplc / openplc
```

Change the default password immediately (Users → openplc). Record that you did
so; the testbed is meant to be reproducible, not insecure.

## 2. Upload the program

`Programs → Choose File → two_tank.st → Upload Program`
Name: `two_tank`, Description: `Phase 1 two-tank level control`.

The file picker runs in the browser, which is on the Windows side, so it cannot
see `~/cps-factory/edge/plc/two_tank.st` in WSL. Either copy it somewhere the
picker can reach (`cp edge/plc/two_tank.st /mnt/d/` then browse to `D:\`) or
use the UNC path `\\wsl$\Ubuntu-22.04\home\<user>\cps-factory\edge\plc\`.

**Uploading is not compiling.** The upload registers the file in OpenPLC's
database; the runtime keeps executing whatever binary was built last. See §4.

## 3. Add the digital twin as a slave device

`Slave Devices → Add new device`

| Field | Value |
|---|---|
| Device Name | `two_tank_twin` |
| Device Type | `Generic Modbus TCP Device` |
| IP Address | `192.168.65.254` — see below |
| IP Port | `5020` |
| Discrete Inputs (%IX100.0) | Start 0, Size 0 |
| Coils (%QX100.0) | Start **0**, Size **3** |
| Input Registers (%IW100) | Start **0**, Size **5** |
| Holding Registers - Read (%IW100) | Start 0, Size 0 |
| Holding Registers - Write (%QW100) | Start 0, Size 0 |

The IP is the Docker Desktop VM's gateway back to the WSL host, **not**
`127.0.0.1`, which from inside the container means the container itself. The
compose file does not use host networking; it publishes ports and sets
`extra_hosts: host.docker.internal:host-gateway`. Confirm the address rather
than assuming it — a Docker Desktop upgrade can move it:

```bash
docker exec cps-openplc getent ahostsv4 host.docker.internal
```

`ahostsv4` forces IPv4. libmodbus accepts dotted quads only, and an IPv6 answer
produces the misleading error `Connection failed on MB device: Invalid argument`.

The resulting address map:

| PLC address | Meaning | Modbus object |
|---|---|---|
| `%IW100` | tank 1 level, mm | input register 0 |
| `%IW101` | tank 2 level, mm | input register 1 |
| `%IW102` | pump inflow, cL/min | input register 2 |
| `%IW103` | V12 flow, cL/min | input register 3 |
| `%IW104` | outlet flow, cL/min | input register 4 |
| `%QX100.0` | pump contactor | coil 0 |
| `%QX100.1` | valve V12 | coil 1 |
| `%QX100.2` | high-high alarm | coil 2 |

> If OpenPLC assigns a different starting index (it depends on how many slave
> devices already exist), read the index shown on the Slave Devices page and
> update the `AT %IW…` declarations in `two_tank.st` to match, then recompile.

## 4. Compile and start

`Settings → Enable Modbus Server (port 502) → Save changes`.

Then compile — this is the step that is easy to miss, because the Programs list
page has no compile button:

`Programs → click the `two_tank` **row** → Program Info → **Launch program**`

Wait for the compiler output to end with `Compilation finished successfully!`,
preceded by `Generating glueVars...` and the eight `__IW100`–`__IW104` /
`__QX100_0`–`__QX100_2` declarations. If those eight are absent or misnamed, the
address map is wrong — see §5.

Compiling leaves the runtime stopped. Finish with `Dashboard → Start PLC`.

## 5. Verify from the plant, not from the dashboard

The web UI is not evidence. Its status line desynchronises from the runtime
core, and it will report a program by name while the runtime executes a
different binary. Verify behaviourally, with the twin running:

```bash
tail -3 logs/twin.log
```

`valve=1` is the first thing to appear — the ST program sets it `TRUE` on every
scan, so it flips within a second of the correct binary running. Then `h1`
should cycle between 250 and 350 mm with `pump` toggling. All coils reading 0 is
the signature of a blank program: registered but never compiled.

Expect a startup transient. On the first scan `H1_MM` reads 0 before the input
registers are populated, which satisfies `H1_MM <= SP_LOW` and latches the pump
on until the level reaches `SP_HIGH`.

## 6. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| All `%IW` read 0 | twin not running, or wrong port | start `twin_server.py`; check `ss -ltn \| grep 5020` |
| All coils read 0, plant uncontrolled | program uploaded but never compiled; runtime is executing the blank program | Programs → click the row → Launch program |
| Levels move but pump never toggles | coils not mapped, or size 0 | set Coils Start 0 / Size 3 and recompile |
| `Invalid argument` on the MB device | libmodbus takes IPv4 literals only; the name resolved to IPv6 | use the dotted quad from `getent ahostsv4` |
| `Connection refused` in the OpenPLC log | slave device pointed at `127.0.0.1`, which is the container itself | use the Docker VM gateway address. Do **not** switch to `network_mode: host` — under Docker Desktop that means the Docker VM, not WSL, and the service becomes unreachable |
| Dashboard says Stopped while the plant is controlled | web layer's status variable desynced from the runtime core | check `docker exec cps-openplc ps aux \| grep core/openplc` and the plant. Do not press Start on a running runtime — it reloads the program and drops the Modbus connection |
| Level drifts to 0 and stays | pump coil never energised | check `SP_LOW`/`SP_HIGH` units — the register is **millimetres** |
| Compile error `unknown located variable` | address index mismatch | align `AT %IW…` with the Slave Devices page |

## 7. What does not persist

The uploaded program, the slave-device entry and the user account live in
OpenPLC's own database inside the container. They survive a `--force-recreate`
only because that database is present in the image; a rebuild from the pinned
submodule produces an image without them, and this procedure must then be
repeated from §2.
