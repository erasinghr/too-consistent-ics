(* =========================================================================
   Hybrid AI-Quantum Cognitive Security Framework Phase 1 - two-tank level control
   Target : OpenPLC Runtime v3 (IEC 61131-3 Structured Text)
   Plant  : edge/twin/twin_server.py, polled as Modbus TCP slave device

   Address map (OpenPLC assigns slave-device I/O from index 100 upwards):
     %IW100  H1_MM   tank 1 level          (mm)
     %IW101  H2_MM   tank 2 level          (mm)
     %IW102  Q_IN    pump inflow           (cL/min)
     %IW103  Q12     valve V12 flow        (cL/min)
     %IW104  Q_OUT   outlet flow           (cL/min)
     %QX100.0 PUMP   pump contactor
     %QX100.1 VALVE  valve V12 solenoid
     %QX100.2 ALARM  high-high alarm lamp

   Control law: two-position (hysteresis) level control on tank 1 between
   SP_LOW and SP_HIGH, with a high-high interlock that trips the pump.
   Keep SP_LOW / SP_HIGH / HH_LIMIT in step with configs/twin.yaml.
   ========================================================================= *)

PROGRAM TwoTankCtrl
  VAR
    H1_MM  AT %IW100 : INT;
    H2_MM  AT %IW101 : INT;
    Q_IN   AT %IW102 : INT;
    Q12    AT %IW103 : INT;
    Q_OUT  AT %IW104 : INT;
    PUMP   AT %QX100.0 : BOOL;
    VALVE  AT %QX100.1 : BOOL;
    ALARM  AT %QX100.2 : BOOL;
  END_VAR

  VAR CONSTANT
    SP_LOW   : INT := 250;   (* pump ON  at or below 250 mm *)
    SP_HIGH  : INT := 350;   (* pump OFF at or above 350 mm *)
    HH_LIMIT : INT := 550;   (* high-high trip level        *)
  END_VAR

  VAR
    HH_TRIP  : BOOL := FALSE;
  END_VAR

  (* --- valve V12 is held open in normal production ------------------- *)
  VALVE := TRUE;

  (* --- high-high interlock (latching until level falls below SP_HIGH) - *)
  IF (H1_MM >= HH_LIMIT) OR (H2_MM >= HH_LIMIT) THEN
    HH_TRIP := TRUE;
  ELSIF (H1_MM < SP_HIGH) AND (H2_MM < SP_HIGH) THEN
    HH_TRIP := FALSE;
  END_IF;
  ALARM := HH_TRIP;

  (* --- two-position level control on tank 1 -------------------------- *)
  IF HH_TRIP THEN
    PUMP := FALSE;
  ELSIF H1_MM <= SP_LOW THEN
    PUMP := TRUE;
  ELSIF H1_MM >= SP_HIGH THEN
    PUMP := FALSE;
  END_IF;

END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK Main(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM Inst0 WITH Main : TwoTankCtrl;
  END_RESOURCE
END_CONFIGURATION
