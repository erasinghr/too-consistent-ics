model TwoTank
  "Two-tank gravity-drained level process. Digital-twin plant model for the
   Hybrid AI-Quantum Cognitive Security Framework CPS testbed (Phase 1). All quantities in SI units.

   Tank 1 is fed by an on/off pump and drains through valve V12 into Tank 2;
   Tank 2 drains to a sink through a fixed restriction. Outflows follow
   Torricelli's law, q = Cv * sqrt(2*g*h), which gives the model the
   first-principles, non-linear dynamics required by the physics term of the
   PN-Fusion detector."

  // ---------------------------------------------------------------- parameters
  parameter Real A1(unit = "m2")     = 0.0154 "Tank 1 cross-sectional area (140 mm dia.)";
  parameter Real A2(unit = "m2")     = 0.0154 "Tank 2 cross-sectional area (140 mm dia.)";
  parameter Real Cv12(unit = "m2")   = 1.2e-5 "Discharge coefficient, valve V12 (tank 1 -> tank 2)";
  parameter Real Cv2o(unit = "m2")   = 1.2e-5 "Discharge coefficient, outlet restriction (tank 2 -> sink)";
  parameter Real Qpump(unit = "m3/s")= 6.0e-5 "Pump volumetric flow when energised (3.6 L/min)";
  parameter Real g(unit = "m/s2")    = 9.81   "Gravitational acceleration";
  parameter Real h1_0(unit = "m")    = 0.30   "Initial level, tank 1";
  parameter Real h2_0(unit = "m")    = 0.30   "Initial level, tank 2";
  parameter Real h_eps(unit = "m")   = 1e-6   "Numerical floor on level inside sqrt()";

  // ------------------------------------------------------- inputs (from the PLC)
  input Real u_pump(start = 0)  "Pump command from PLC coil, 0 or 1";
  input Real u_valve(start = 1) "Valve V12 command from PLC coil, 0 or 1";

  // ------------------------------------------------------------------- outputs
  output Real h1(unit = "m", start = h1_0, fixed = true) "Level, tank 1";
  output Real h2(unit = "m", start = h2_0, fixed = true) "Level, tank 2";
  output Real q_in(unit = "m3/s")  "Pump inflow to tank 1";
  output Real q12(unit = "m3/s")   "Flow tank 1 -> tank 2 through V12";
  output Real q_out(unit = "m3/s") "Outflow from tank 2 to sink";

equation
  q_in  = Qpump * u_pump;
  q12   = u_valve * Cv12 * sqrt(2 * g * noEvent(max(h1, h_eps)));
  q_out = Cv2o * sqrt(2 * g * noEvent(max(h2, h_eps)));

  der(h1) = (q_in - q12) / A1;
  der(h2) = (q12 - q_out) / A2;

  annotation(
    experiment(StartTime = 0, StopTime = 600, Interval = 0.1, Tolerance = 1e-6),
    Documentation(info = "<html><p>Plant model for Phase 1 of the Hybrid AI-Quantum Cognitive Security Framework CPS
    security framework. Exported as an FMI 2.0 co-simulation FMU and driven by
    <code>twin_server.py</code>, which exposes the levels as Modbus input
    registers and accepts the PLC actuator commands as Modbus coils.</p></html>"));
end TwoTank;
