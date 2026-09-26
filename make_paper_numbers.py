"""Turn results/summary.json into LaTeX macros and tables so the paper can never drift from the code."""
import json
import os
import re

import numpy as np

from engine import SimConfig, build_ephemeris, generate_workload

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "paper")
S = json.load(open(os.path.join(HERE, "results", "summary.json")))
M = S["main"]
days = S["days"]
EL, RB, GP, SF, AT = (M[k] for k in ("E-LOTUS", "Rule-Based", "Greedy-Process", "Store-and-Forward", "Always-Transmit"))

# runtime of the main E-LOTUS run (from the benchmark log)
runtime = float("nan")
log = open(os.path.join(HERE, "results_full.log")).read()
mm = re.search(r"\[main\s*\]\s+E-LOTUS\s+([\d.]+)s", log)
if mm:
    runtime = float(mm.group(1))

cfg = SimConfig(seed=S["main_seed"], duration_h=24.0 * days)
eph = build_ephemeris(cfg)
from engine import OrbitPropagator, read_tle
from sgp4.api import Satrec
_tle = read_tle(cfg)
_sat = Satrec.twoline2rv(_tle[1], _tle[2])
period_min = 2 * np.pi / _sat.no_kozai
n24 = int(86400 / cfg.dt)
_st = OrbitPropagator.propagate(tuple(eph.r[0]), tuple(eph.v[0]), cfg.dt, n24 + 1)
rk4_err_km = float(np.linalg.norm(_st[n24, :3] - eph.r[n24]))
RD = json.load(open(os.path.join(HERE, "results", "realdata_stats.json")))
arr = generate_workload(cfg, eph)
raw_mb = sum(t.size_mb for slot in arr for t in slot)
n_slots = cfg.n_slots
sunlit_pct = 100 * eph.sunlit[:n_slots].mean()
dl_gb_day = EL["contact_min_per_day"] * 60 * cfg.dl_mb_per_s / 1000
raw_gb_day = raw_mb / days / 1000
demand_duty = 100 * (raw_mb / days) * cfg.gflop_per_mb / cfg.gpu_gflops / 86400


def f(x, d=1):
    return f"{x:.{d}f}"


macros = {
    "Days": str(days),
    "ELvalueDay": f(EL["value"]), "RBvalueDay": f(RB["value"]), "GPvalueDay": f(GP["value"]),
    "SFvalueDay": f(SF["value"]), "ATvalueDay": f(AT["value"]),
    "ELceil": f(EL["ceiling_pct"]), "RBceil": f(RB["ceiling_pct"]),
    "ELcompl": f(EL["completion_pct"]), "RBcompl": f(RB["completion_pct"]),
    "ELcomplDelta": f(EL["completion_pct"] - RB["completion_pct"]),
    "ELvalueGainRB": f(100 * (EL["value"] / RB["value"] - 1)),
    "ELvalueGainSF": f(EL["value"] / SF["value"]),
    "ELvalueGainAT": f(EL["value"] / AT["value"], 0),
    "ELenergy": f(EL["energy_payload_wh"]), "RBenergy": f(RB["energy_payload_wh"]),
    "ELenergyRatio": f(100 * EL["energy_payload_wh"] / RB["energy_payload_wh"]),
    "ELenergyDeltaPct": f(100 * (EL["energy_payload_wh"] / RB["energy_payload_wh"] - 1)),
    "GPshed": f(GP["shed_pct"]), "GPtmax": f(GP["temp_max_c"]),
    "ELsocmin": f(EL["soc_min_pct"], 0), "RBsocmin": f(RB["soc_min_pct"], 0), "GPsocmin": f(GP["soc_min_pct"], 0),
    "ELtmax": f(EL["temp_max_c"]), "RBtmax": f(RB["temp_max_c"]),
    "ELlat": f(EL["latency_mean_min"], 0), "RBlat": f(RB["latency_mean_min"], 0), "SFlat": f(SF["latency_mean_min"], 0),
    "ELovf": f(EL["overflow"], 0), "RBovf": f(RB["overflow"], 0),
    "ELovfRed": f(100 * (1 - EL["overflow"] / RB["overflow"]), 0),
    "ELoffl": f(EL["offloaded"], 0), "ELalerts": f(EL["alerts_on_time_pct"]), "RBalerts": f(RB["alerts_on_time_pct"]),
    "SFalerts": f(SF["alerts_on_time_pct"]),
    "ELgpu": f(EL["gpu_duty_pct"]), "ELdl": f(EL["dl_util_pct"]), "ELvpw": f(EL["value_per_wh"], 2),
    "RBvpw": f(RB["value_per_wh"], 2), "SFvpw": f(SF["value_per_wh"], 2),
    "ContactMin": f(EL["contact_min_per_day"], 0), "SunlitPct": f(sunlit_pct), "SlotsK": f(n_slots / 1000, 0),
    "ELruntime": f(runtime, 0), "ELmsPerSlot": f(1000 * runtime / n_slots, 2),
    "GenPerDay": f(EL["generated"], 0), "RawGBDay": f(raw_gb_day, 0), "DlGBDay": f(dl_gb_day, 1),
    "DemandDuty": f(demand_duty, 0), "CeilDay": f(EL["ceiling_value"]),
    "PeriodMin": f(period_min, 2), "AltMean": f(float(eph.alt[:n_slots].mean()), 0),
    "AltMin": f(float(eph.alt[:n_slots].min()), 0), "AltMax": f(float(eph.alt[:n_slots].max()), 0),
    "EclMax": f(float(eph.ecl_remaining_s[:n_slots].max()) / 60, 1), "BetaDeg": f(float(np.median(eph.beta_deg[:n_slots])), 0),
    "RKerr": f(rk4_err_km, 1), "TleInc": f(np.degrees(_sat.inclo), 2), "TleEpoch": "2026-09-26",
    "RhoRawOpt": f(RD["rho_raw_optical"], 2), "RhoRawTir": f(RD["rho_raw_tir"] if abs(RD["rho_raw_tir"]) >= 0.005 else 0.0, 2), "AlertRate": f(RD["alert_rate_pct"], 1),
    "RhoWithinOpt": f(RD["within_land_rho_H_clear"]["optical"], 2), "RhoWithinTir": f(RD["within_land_rho_H_clear"]["tir"], 2),
    "CapRandom": f(100 * RD["heldout_capture_top30"]["random"], 0), "CapRaw": f(100 * RD["heldout_capture_top30"]["raw entropy"], 0),
    "CapLand": f(100 * RD["heldout_capture_top30"]["land prior"], 0), "CapEL": f(100 * RD["heldout_capture_top30"]["E-LOTUS"], 0),
    "CapOracle": f(100 * RD["heldout_capture_top30"]["oracle"], 0), "RhoEL": f(RD["heldout_rho"]["E-LOTUS"], 2),
    "ShareDark": f(RD["share_by_class_pct"]["dark"], 0),
    "ShareOcean": f(RD["share_by_class_pct"]["ocean"], 0), "ShareCloud": f(RD["share_by_class_pct"]["cloud"], 0),
    "ShareCoast": f(RD["share_by_class_pct"]["coast"], 0), "ShareLand": f(RD["share_by_class_pct"]["land"], 0),
    "ShareFire": f(RD["share_by_class_pct"]["fire"], 1),
    "HOcean": f(RD["H_by_class_optical"]["ocean"][0], 1), "HCloud": f(RD["H_by_class_optical"]["cloud"][0], 1),
    "HCoast": f(RD["H_by_class_optical"]["coast"][0], 1), "HLand": f(RD["H_by_class_optical"]["land"][0], 1),
    "HFire": f(RD["H_by_class_optical"]["fire"][0], 1), "AtlasN": str(RD["n"]), "FireThr": str(cfg.fire_trigger_px),
    "DateStart": RD["dates"][0], "DateEnd": RD["dates"][1], "HalfDays": str(RD["days"] // 2),
}
# ablations
for key, tag in (("E-LOTUS w/o ISLL", "NoISL"), ("E-LOTUS w/o entropy", "NoEnt"), ("E-LOTUS static E_min", "Static"),
                 ("E-LOTUS raw entropy", "Raw"),
                 ("E-LOTUS static E_min V=0.01", "StaticLowV")):
    a = M[key]
    macros[f"{tag}value"] = f(a["value"])
    macros[f"{tag}valueDelta"] = f(100 * (a["value"] / EL["value"] - 1))
    macros[f"{tag}compl"] = f(a["completion_pct"])
    macros[f"{tag}socmin"] = f(a["soc_min_pct"], 0)
    macros[f"{tag}tmax"] = f(a["temp_max_c"])
    macros[f"{tag}energy"] = f(a["energy_payload_wh"])
    macros[f"{tag}dl"] = f(a["dl_util_pct"])
    macros[f"{tag}shed"] = f(a["shed_pct"])
    macros[f"{tag}ovf"] = f(a["overflow"], 0)
    macros[f"{tag}lat"] = f(a["latency_mean_min"], 0)
LV = S["vsweep"]["0.01"]
sl = M["E-LOTUS static E_min V=0.01"]
macros["StaticLowVvsLowV"] = f(100 * (sl["value"] / LV["value"] - 1))
macros["LowVcompl"] = f(LV["completion_pct"])
macros.update({"LowVvalue": f(LV["value"]), "LowVsocmin": f(LV["soc_min_pct"], 0), "LowVshed": f(LV["shed_pct"]),
               "LowVenergy": f(LV["energy_payload_wh"])})
vs = sorted(S["vsweep"].items(), key=lambda kv: float(kv[0]))
net = [m["value"] - 0.5 * m["energy_payload_wh"] for _, m in vs]
macros.update({"VnetLo": f(net[0]), "VnetHi": f(max(net)), "VqLo": f(vs[0][1]["backlog_mean_mb"] / 1000, 2),
               "VqHi": f(vs[-1][1]["backlog_mean_mb"] / 1000, 2),
               "VsocLo": f(min(m["soc_min_pct"] for V, m in vs if float(V) <= 0.1), 0),
               "VsocHi": f(min(m["soc_min_pct"] for V, m in vs if float(V) >= 0.3), 0)})
# multi-seed
SE = S["seeds"]
e, r = SE["E-LOTUS"], SE["Rule-Based"]
macros.update({"MSelValue": f(e["value"][0]), "MSelValueStd": f(e["value"][1]), "MSrbValue": f(r["value"][0]),
               "MSrbValueStd": f(r["value"][1]),
               "MSgain": f(100 * (e["value"][0] / r["value"][0] - 1)),
               "MScomplDelta": f(e["completion_pct"][0] - r["completion_pct"][0]),
               "MSenergyDelta": f(100 * (e["energy_payload_wh"][0] / r["energy_payload_wh"][0] - 1)),
               "MSlatDelta": f(e["latency_mean_min"][0] - r["latency_mean_min"][0], 0)})
def word(x, unit, less, more, d=1):
    return f"{abs(x):.{d}f}{unit} {less if x < 0 else more}"


macros["MSenergyWord"] = word(100 * (e["energy_payload_wh"][0] / r["energy_payload_wh"][0] - 1), "\\%", "less", "more")
macros["MSlatWord"] = word(e["latency_mean_min"][0] - r["latency_mean_min"][0], "~min", "lower", "higher", 0)
macros["NoEntdrop"] = f(-100 * (M["E-LOTUS w/o entropy"]["value"] / EL["value"] - 1))
macros["Rawdrop"] = f(-100 * (M["E-LOTUS raw entropy"]["value"] / EL["value"] - 1))
macros["RawGainEL"] = f(100 * (EL["value"] / M["E-LOTUS raw entropy"]["value"] - 1))
macros["NoISLdrop"] = f(-100 * (M["E-LOTUS w/o ISLL"]["value"] / EL["value"] - 1))
macros["StaticLowVdrop"] = f(-100 * (M["E-LOTUS static E_min V=0.01"]["value"] / LV["value"] - 1))
ST = S["stress"]
wins = sum(1 for sc, d in ST.items() if d["E-LOTUS"]["value"][0] > d["Rule-Based"]["value"][0])
macros.update({"StressWins": str(wins), "StressN": str(len(ST))})
gains = [100 * (d["E-LOTUS"]["value"][0] / d["Rule-Based"]["value"][0] - 1) for d in ST.values()]
cgains = [d["E-LOTUS"]["completion_pct"][0] - d["Rule-Based"]["completion_pct"][0] for d in ST.values()]
macros.update({"StressGainMin": f(min(gains)), "StressGainMax": f(max(gains)),
               "StressComplMin": f(min(cgains)), "StressComplMax": f(max(cgains))})
L = ST["load x1.5"]
macros.update({"LoadELsoc": f(L["E-LOTUS"]["soc_min_pct"][0], 0), "LoadRBsoc": f(L["Rule-Based"]["soc_min_pct"][0], 0),
               "LoadGPvalue": f(L["Greedy-Process"]["value"][0]), "LoadELvalue": f(L["E-LOTUS"]["value"][0])})
R = ST["radiator x0.75"]
macros.update({"RadGPthr": f(R["Greedy-Process"]["throttled_pct"][0]), "RadELtmax": f(R["E-LOTUS"]["temp_max_c"][0]),
               "RadRBtmax": f(R["Rule-Based"]["temp_max_c"][0])})

with open(os.path.join(P, "numbers.tex"), "w") as fh:
    fh.write("% auto-generated by make_paper_numbers.py -- do not edit\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- Table I
c = SimConfig()
rows = [
    ("Orbit", "PHISAT-2 (NORAD 60470), CelesTrak TLE epoch 2026-09-26, SGP4, $\\Delta t=10$ s"),
    ("Altitude / period", "\\AltMin--\\AltMax\\ km, $i=\\TleInc^\\circ$, $T=\\PeriodMin$ min, eclipse $\\le$\\EclMax\\ min"),
    ("Ground stations", "KSAT SvalSat, ASI Matera; mask $10^\\circ$; \\ContactMin\\ min/day"),
    ("Earth data", "NASA GIBS daily VIIRS/MODIS, \\DateStart\\ to \\DateEnd, 313 km quick-look blocks"),
    ("Battery", f"{c.battery_wh:.0f} Wh LiFePO$_4$, $\\eta_c=\\eta_d={c.eta_charge}$, $P_{{\\rm ch}}\\le{c.p_charge_max_w:.0f}$ W"),
    ("Floors", f"$E_{{\\rm crit}}={c.e_crit_wh:.0f}$ Wh, $E_{{\\rm res}}={c.e_reserve_wh:.0f}$ Wh (+ eclipse term)"),
    ("Solar array", f"3 body faces $\\times$ {c.p_face_w:.0f} W at normal incidence"),
    ("Loads", f"bus {c.p_bus_w:.0f} W; GPU {c.p_gpu_idle_w}/{c.p_gpu_active_w:.0f} W; TX {c.p_tx_w:.0f} W; ISL {c.p_isl_w:.0f} W"),
    ("Compute", f"Jetson Orin-class, {c.gpu_gflops:.0f} GFLOP/s eff.; {c.gflop_per_mb:.0f} GFLOP/MB"),
    ("Product", f"$\\kappa={c.product_ratio}$, $\\eta_p={c.eta_process}$"),
    ("Radiator", f"$\\epsilon={c.emissivity}$, $A={c.radiator_area_m2}$ m$^2$, $C={c.heat_capacity_j_per_k:.0f}$ J/K"),
    ("Thermal limits", f"$T_{{\\rm safe}}={c.t_safe_c:.0f}^\\circ$C (soft), $T_{{\\max}}={c.t_max_c:.0f}^\\circ$C (throttle)"),
    ("Env. heat", f"$Q_{{\\rm sun}}={c.q_sun_w:.0f}$ W (sunlit), $Q_{{\\rm IR}}={c.q_earth_ir_w:.0f}$ W"),
    ("Links", f"S/X {c.downlink_mbps:.0f} Mbps; optical ISL {c.isl_mbps:.0f} Mbps, neighbours $\\pm{c.isl_phase_deg:.0f}^\\circ$"),
    ("Donor model", f"{c.nb_spare_gflops:.0f} GFLOP/s spare, {c.nb_bucket_gflop/1000:.0f} TFLOP bucket"),
    ("Storage", f"{c.storage_mb/1000:.0f} GB ring buffer"),
    ("Deadlines", "alert 90 min, routine 12 h"),
    ("E-LOTUS", "$V=0.3$, $\\zeta=0.5$, $\\alpha=3$, $\\varepsilon=0.1$, $\\Delta E=5$ Wh, $\\Delta T=5$ K"),
]
with open(os.path.join(P, "table_params.tex"), "w") as fh:
    fh.write("\\begin{table}[t]\n\\centering\n\\caption{Simulation parameters}\\label{tab:params}\n\\footnotesize\n"
             "\\begin{tabular}{@{}p{0.24\\columnwidth}p{0.70\\columnwidth}@{}}\n\\toprule\n")
    for a, b in rows:
        fh.write(f"{a} & {b}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- Table II
order = ["E-LOTUS", "Rule-Based", "Store-and-Forward", "Always-Transmit", "Greedy-Process"]
cols = [("value", "Value/day", 1), ("ceiling_pct", "\\% ceil.", 1), ("completion_pct", "Compl.\\,\\%", 1),
        ("alerts_on_time_pct", "Alerts\\,\\%", 1), ("energy_payload_wh", "Energy Wh/d", 0),
        ("value_per_wh", "Value/Wh", 2), ("latency_mean_min", "Lat.\\,min", 0), ("soc_min_pct", "SOC$_{\\min}$\\,\\%", 0),
        ("temp_max_c", "$T_{\\max}$\\,$^\\circ$C", 1), ("shed_pct", "Shed\\,\\%", 1), ("gpu_duty_pct", "GPU\\,\\%", 0),
        ("dl_util_pct", "DL\\,\\%", 0), ("overflow", "Ovf./d", 0)]
best_hi = {"value", "ceiling_pct", "completion_pct", "alerts_on_time_pct", "value_per_wh"}
with open(os.path.join(P, "table_results.tex"), "w") as fh:
    fh.write("\\begin{table*}[t]\n\\centering\n\\caption{%d-day benchmark (seed %d): per-day averages. "
             "Ceiling = contact-limited upper bound (%s value/day). Best in bold.}\\label{tab:main}\n\\footnotesize\n"
             "\\setlength{\\tabcolsep}{3.1pt}\n\\begin{tabular}{@{}l%s@{}}\n\\toprule\n" % (days, S["main_seed"], f(EL["ceiling_value"]), "r" * len(cols)))
    fh.write("Controller & " + " & ".join(h for _, h, _ in cols) + "\\\\\n\\midrule\n")
    for p in order:
        cells = []
        for k, _, d in cols:
            v = M[p][k]
            s = f(v, d)
            if k in best_hi and abs(v - max(M[q][k] for q in order)) < 1e-9:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        name = "\\textbf{E-LOTUS (ours)}" if p == "E-LOTUS" else p
        fh.write(name + " & " + " & ".join(cells) + "\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---------------------------------------------------------------- Table III ablations
ab = [("E-LOTUS (full, $V$=0.3)", "E-LOTUS"), ("\\quad w/o ISLL offload", "E-LOTUS w/o ISLL"),
      ("\\quad w/o entropy weighting", "E-LOTUS w/o entropy"), ("\\quad raw entropy only", "E-LOTUS raw entropy"),
      ("\\quad static floors", "E-LOTUS static E_min"),
      ("E-LOTUS, $V$=0.01", None), ("\\quad static floors, $V$=0.01", "E-LOTUS static E_min V=0.01")]
acols = [("value", "Value/d", 1), ("completion_pct", "Compl.\\%", 1), ("energy_payload_wh", "Wh/d", 0),
         ("soc_min_pct", "SOC$_{\\min}$", 0), ("temp_max_c", "$T_{\\max}$", 1), ("shed_pct", "Shed\\%", 1)]
with open(os.path.join(P, "table_ablation.tex"), "w") as fh:
    fh.write("\\begin{table}[t]\n\\centering\n\\caption{Ablations (%d days, per-day averages)}\\label{tab:abl}\n\\footnotesize\n"
             "\\setlength{\\tabcolsep}{1.6pt}\n\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n" % days)
    fh.write("Variant & " + " & ".join(h for _, h, _ in acols) + "\\\\\n\\midrule\n")
    for name, key in ab:
        m = S["vsweep"]["0.01"] if key is None else M[key]
        fh.write(name + " & " + " & ".join(f(m[k], d) for k, _, d in acols) + "\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- Table IV robustness
scen = [("Nominal (5 seeds)", None)] + [(k, k) for k in ["load x1.5", "radiator x0.75", "SOC0 35%", "storage 4 GB", "Svalbard only"]]
pretty = {"load x1.5": "Load $\\times1.5$", "radiator x0.75": "Radiator $k=0.75$", "SOC0 35%": "Initial SOC 35\\%",
          "storage 4 GB": "Storage 4 GB", "Svalbard only": "Svalbard only"}
rcols = [("value", "Value", 1), ("ceiling_pct", "\\% ceil.", 1), ("completion_pct", "Compl.\\,\\%", 1),
         ("energy_payload_wh", "Energy Wh", 0), ("soc_min_pct", "SOC$_{\\min}$\\,\\%", 0), ("temp_max_c", "$T_{\\max}$\\,$^\\circ$C", 1),
         ("shed_pct", "Shed\\,\\%", 1), ("throttled_pct", "Thr.\\,\\%", 1)]
with open(os.path.join(P, "table_stress.tex"), "w") as fh:
    fh.write("\\begin{table*}[t]\n\\centering\n\\caption{Robustness over 24-h horizons: mean $\\pm$ std over independent seeds "
             "(5 seeds nominal, 3 seeds per stress scenario)}\\label{tab:stress}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.0pt}\n"
             "\\begin{tabular}{@{}ll%s@{}}\n\\toprule\n" % ("r" * len(rcols)))
    fh.write("Scenario & Controller & " + " & ".join(h for _, h, _ in rcols) + "\\\\\n\\midrule\n")
    for label, key in scen:
        d = SE if key is None else ST[key]
        pols = ["E-LOTUS", "Rule-Based", "Greedy-Process"]
        for j, p in enumerate(pols):
            m = d[p]
            first = (("\\multirow{3}{*}{" + (label if key is None else pretty[key]) + "}") if j == 0 else "")
            cells = [f"{f(m[k][0], dd)}$\\pm${f(m[k][1], dd)}" if k in ("value", "ceiling_pct", "completion_pct") else f(m[k][0], dd)
                     for k, _, dd in rcols]
            fh.write(f"{first} & {p} & " + " & ".join(cells) + "\\\\\n")
        fh.write("\\midrule\n" if label != scen[-1][0] else "")
    fh.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

print("\n".join(f"{k} = {v}" for k, v in macros.items()))
