"""Pre-compute every scenario shown on the Orbitra web page (all numbers come from engine.py).

* replay grid   : E-LOTUS (V x SOC0 x radiator k x storage x ISL) + Greedy / Rule-Based (SOC0 x k x storage x ISL)
                  on the real orbit, first real day, 24 h, 1-min telemetry + E-LOTUS decision log
* orbit grid    : inclination x LTAN what-if orbits (RK4 two-body + J2, nearest real NASA scenes) + the real orbit;
                  daily metrics, 5-min battery curve and one representative orbit (1-min) for the 3D intro
Run:  python web/build_web_data.py        (about 6 min on 2 cores)
"""
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine import (ACTIONS, ELotusEngine, GreedyProcess, RuleBased, SimConfig, Simulation,  # noqa: E402
                    build_ephemeris, generate_workload)

SEED, DAY = 2026, 0
V_SET = [0.03, 0.3, 3.0]
SOC_SET = [35, 70, 100]
K_SET = [0.75, 1.0, 1.25]
ST_SET = [4, 8, 16]
ISL_SET = [1, 0]
INC_SET = [45.0, 60.0, 75.0, 90.0, 97.4]
LTAN_SET = [x * 1.5 for x in range(16)]

_W = {}


def world(key, **kw):
    if key not in _W:
        cfg = SimConfig(seed=SEED, start_day=DAY, duration_h=24.0, **kw)
        eph = build_ephemeris(cfg)
        _W[key] = (cfg, eph, generate_workload(cfg, eph))
    return _W[key]


def minute_actions(act, k=6):
    out = []
    for i in range(0, len(act), k):
        seg = act[i:i + k]
        out.append(int(3 if (seg == 3).any() else 2 if (seg == 2).any() else 1 if (seg == 1).any() else 0))
    return out


METRIC_KEYS = ["value", "ceiling_pct", "completion_pct", "alerts_on_time_pct", "energy_payload_wh",
               "latency_mean_min", "soc_min_pct", "temp_max_c", "shed_pct", "throttled_pct", "offloaded"]


def run_replay(job):
    pol, V, soc, k, st, isl = job
    cfg, eph, arr = world(("r", soc, k, st, isl), soc0=soc / 100, k_thermal=k, storage_mb=st * 1000.0,
                          isl_enabled=bool(isl))
    p = {"E-LOTUS": lambda: ELotusEngine(V=V), "Greedy-Process": GreedyProcess, "Rule-Based": RuleBased}[pol]()
    r = Simulation(cfg, p, eph, arr).run()
    t = r.telemetry
    idx = np.arange(0, len(t["t_h"]), 6)
    flags = minute_actions(t["action"].astype(int))
    out = dict(
        soc=np.round(t["soc"][idx] * 2).astype(int).tolist(),                 # 0.5 % units
        temp=np.round((t["temp_c"][idx] + 20) * 4).astype(int).tolist(),       # 0.25 K units, offset -20 C
        q=np.round(t["q_mb"][idx] / 50).astype(int).tolist(),                  # 50 MB units
        val=np.round(t["value_cum"][idx] * 10).astype(int).tolist(),           # 0.1 units
        f=[a | (int(t["shed"][i]) << 2) | (int(t["throttled"][i]) << 3) for a, i in zip(flags, idx)],
        m={kk: round(float(r.metrics[kk]), 2) for kk in METRIC_KEYS})
    if pol == "E-LOTUS":
        ev = [e for e in r.events if e["detail"] != "arrival"]
        out["ev_t"] = [int(round(e["t_h"] * 360)) for e in ev]                  # 10-s slots
        out["ev_i"] = [int(e["task_id"]) for e in ev]
        out["ev_c"] = [ACTIONS.index(e["action"]) | ({"": 0, "lead": 1, "trail": 2}.get(e["detail"], 0) << 2) for e in ev]
    return job, out


def orbit_window(eph, n_day, dt):
    """One representative orbit: starts at an ascending node between 6 h and 20 h, most ground contact."""
    lat = eph.lat[:n_day]
    asc = np.flatnonzero((lat[:-1] < 0) & (lat[1:] >= 0))
    period = int(round(np.median(np.diff(asc)))) if len(asc) > 2 else int(5640 / dt)
    best, bs = None, -1
    for s in asc:
        if s * dt < 6 * 3600 or s * dt > 20 * 3600 or s + period >= n_day:
            continue
        c = eph.contact[s:s + period].sum()
        if c > bs:
            best, bs = s, c
    return int(best if best is not None else asc[len(asc) // 2]), period


def run_orbit(job):
    inc, ltan, alt = job
    if inc is None:
        cfg, eph, arr = world(("real",))
    else:
        cfg, eph, arr = world(("o", inc, ltan), orbit_model="rk4_j2", inclination_deg=inc, ltan_h=ltan, altitude_km=alt)
    rel = Simulation(cfg, ELotusEngine(), eph, arr).run()
    rb = Simulation(cfg, RuleBased(), eph, arr).run()
    t = rel.telemetry
    N = cfg.n_slots
    s0, per = orbit_window(eph, N, cfg.dt)
    ii = np.arange(s0, s0 + per + 1, 6)
    r = eph.r[ii] / np.linalg.norm(eph.r[ii], axis=1)[:, None]
    from engine import gmst_rad
    jd0 = cfg.epoch_jd + s0 * cfg.dt / 86400.0
    acts = minute_actions(t["action"].astype(int)[s0:s0 + per + 6])[:len(ii)]
    # which task is being handled: nearest scene before each minute
    win = dict(
        t0_min=int(s0 * cfg.dt / 60), gmst0=float(gmst_rad(np.array([jd0]))[0]),
        sun=[round(float(x), 5) for x in eph.sun[s0]],
        xyz=np.round(r * 10000).astype(int).tolist(),
        lat=np.round(eph.lat[ii], 2).tolist(), lon=np.round(eph.lon[ii], 2).tolist(),
        sunlit=eph.sunlit[ii].astype(int).tolist(), gs=eph.contact_station[ii].astype(int).tolist(),
        soc=np.round(t["soc"][ii], 1).tolist(), temp=np.round(t["temp_c"][ii], 1).tolist(),
        psol=np.round(t["p_solar_w"][ii], 1).tolist(), pload=np.round(t["p_load_w"][ii], 1).tolist(),
        q=np.round(t["q_mb"][ii] / 1000, 2).tolist(), act=acts,
        isl=[int(any(eph.isl_los[s][i] and eph.nb_sunlit[s][i] for s in ("lead", "trail"))) for i in ii],
    )
    m = {kk: round(float(rel.metrics[kk]), 2) for kk in METRIC_KEYS}
    m["rb_value"] = round(float(rb.metrics["value"]), 2)
    m["eclipse_pct"] = round(100 * (1 - eph.sunlit[:N].mean()), 1)
    m["contact_min"] = round(float(eph.contact[:N].sum() * cfg.dt / 60), 1)
    m["solar_wh"] = round(float(t["p_solar_w"].sum() * cfg.dt / 3600), 1)
    m["beta_deg"] = round(float(eph.beta_deg[:N].mean()), 1)
    m["alt_km"] = round(float(eph.alt[:N].mean()), 1)
    m["inc_deg"] = round(float(np.degrees(np.arccos(np.clip(
        np.cross(eph.r[0], eph.v[0])[2] / np.linalg.norm(np.cross(eph.r[0], eph.v[0])), -1, 1)))), 2)
    soc24 = np.round(t["soc"][::30]).astype(int).tolist()                       # 5-min
    return job, dict(m=m, soc24=soc24, win=win)


if __name__ == "__main__":
    t0 = time.time()
    cfg0, eph0, arr0 = world(("real",))
    alt = round(float(eph0.alt[:cfg0.n_slots].mean()))
    replay_jobs = [("E-LOTUS", V, s, k, st, i) for V, s, k, st, i in itertools.product(V_SET, SOC_SET, K_SET, ST_SET, ISL_SET)]
    replay_jobs += [(p, 0.0, s, k, st, i) for p in ("Greedy-Process", "Rule-Based")
                    for s, k, st, i in itertools.product(SOC_SET, K_SET, ST_SET, ISL_SET)]
    orbit_jobs = [(None, None, alt)] + [(inc, lt, alt) for inc in INC_SET for lt in LTAN_SET]
    with Pool(2) as pool:
        orb = dict(pool.map(run_orbit, orbit_jobs, chunksize=2))
        print(f"orbit grid done {time.time() - t0:.0f}s", flush=True)
        rep = dict(pool.map(run_replay, replay_jobs, chunksize=4))
        print(f"replay grid done {time.time() - t0:.0f}s", flush=True)

    # shared real-orbit track, scenes, stations and benchmark summaries
    N = cfg0.n_slots
    ii = np.arange(0, N, 6)
    scenes = [tk for s in arr0 for tk in s]
    S = json.load(open("results/summary.json"))
    RD = json.load(open("results/realdata_stats.json"))
    keys = METRIC_KEYS
    out = dict(
        grid=dict(V=V_SET, soc=SOC_SET, k=K_SET, st=ST_SET, isl=ISL_SET, inc=INC_SET, ltan=LTAN_SET, alt=alt),
        replay={"|".join(map(str, j[:1] + ((j[1],) if j[0] == "E-LOTUS" else ()) + j[2:])): v for j, v in rep.items()},
        orbits={("real" if j[0] is None else f"{j[0]}|{j[1]}"): v for j, v in orb.items()},
        track=dict(lat=np.round(eph0.lat[ii], 2).tolist(), lon=np.round(eph0.lon[ii], 2).tolist(),
                   sunlit=eph0.sunlit[ii].astype(int).tolist(), contact=eph0.contact[ii].astype(int).tolist()),
        scenes=[[round(tk.lat, 2), round(tk.lon, 2), round(tk.entropy, 2), int(tk.priority == 4), tk.scene,
                 round(tk.t_gen / 3600, 3)] for tk in scenes],
        gs={k: v for k, v in cfg0.ground_stations.items()},
        date=json.load(open("results/summary.json")).get("date0", "2026-08-26"),
        main={p: {k: round(S["main"][p][k], 2) for k in keys if k in S["main"][p]} for p in S["main"]},
        stress={s: {p: {k: round(v[k][0], 2) for k in ["value", "completion_pct", "soc_min_pct", "temp_max_c",
                                                        "throttled_pct", "shed_pct"]} for p, v in d.items()}
                for s, d in S["stress"].items()},
        seeds={p: {k: [round(v[k][0], 2), round(v[k][1], 2)] for k in ["value", "completion_pct"]}
               for p, v in S["seeds"].items()},
        real=RD, days=S["days"])
    path = os.path.join(HERE, "orbitra_data.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.2f} MB in {time.time() - t0:.0f}s")
