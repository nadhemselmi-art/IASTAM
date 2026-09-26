"""
E-LOTUS benchmark suite (produces every number and figure used in the paper).

  python benchmark.py            # full suite (~10 min on 2 cores)
  python benchmark.py --quick    # 3-day main run, fewer V points (smoke test)

Outputs
  results/main_30d.csv           30-day comparison, all controllers + ablations
  results/vsweep_30d.csv         Lyapunov trade-off parameter sweep
  results/multiseed_24h.csv      5 independent real days x seeds, 24 h, mean/std
  results/stress_24h.csv         robustness scenarios
  results/summary.json           everything above in one file
  paper/figures/*.png            figures
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from engine import (SimConfig, build_ephemeris, generate_workload, Simulation, ELotusEngine,
                    AlwaysTransmit, GreedyProcess, StoreAndForward, RuleBased, ACTIONS)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "paper", "figures")
MAIN_SEED = 2026          # evaluation trace (V was selected on seed 7, never reused here)

POLICY_COLORS = {"E-LOTUS": "#2a78d6", "Always-Transmit": "#eb6834", "Greedy-Process": "#1baf7a",
                 "Store-and-Forward": "#eda100", "Rule-Based": "#e87ba4"}
ACTION_COLORS = {"STORE": "#c9c8c1", "PROCESS": "#4a3aa7", "TRANSMIT": "#eb6834", "OFFLOAD_ISLL": "#1baf7a"}

KEYS = ["value", "ceiling_value", "ceiling_pct", "value_capture_pct", "completion_pct", "alerts_on_time_pct",
        "energy_payload_wh", "energy_total_wh", "value_per_wh", "latency_mean_min", "latency_p95_min",
        "aoi_mean_min", "soc_min_pct", "soc_mean_pct", "shed_pct", "below_emin_pct", "temp_max_c",
        "above_tsafe_pct", "throttled_pct", "gpu_duty_pct", "dl_util_pct", "isl_gb", "overflow", "expired",
        "processed", "offloaded", "tx_raw", "tx_prod", "generated", "delivered", "backlog_mean_mb",
        "contact_min_per_day"]


def make_policy(spec):
    kind, kw = spec
    return {"elotus": ELotusEngine, "at": AlwaysTransmit, "gp": GreedyProcess,
            "sf": StoreAndForward, "rb": RuleBased}[kind](**kw)


_CACHE = {}


def _run(job):
    """job = (tag, cfg_kwargs, policy_spec, keep_telemetry)"""
    tag, cfg_kw, spec, keep = job
    cfg = SimConfig(**cfg_kw)
    key = tuple(sorted((k, str(v)) for k, v in cfg_kw.items()))
    if key not in _CACHE:
        eph = build_ephemeris(cfg)
        _CACHE.clear()
        _CACHE[key] = (eph, generate_workload(cfg, eph))
    eph, arr = _CACHE[key]
    t0 = time.time()
    res = Simulation(cfg, make_policy(spec), eph, arr, max_events=2000).run()
    out = {"tag": tag, "policy": res.policy, "cfg": cfg_kw, "runtime_s": time.time() - t0,
           "metrics": {k: float(res.metrics[k]) for k in KEYS}}
    if keep:
        tel = res.telemetry
        out["tel"] = {k: tel[k].tolist() for k in ("t_h", "soc", "E_min_wh", "temp_c", "q_mb", "action",
                                                    "sunlit", "contact", "value_cum", "energy_wh_cum", "gpu_util")}
    print(f"  [{tag:10s}] {res.policy:22s} {out['runtime_s']:6.1f}s  value={out['metrics']['value']:8.1f}", flush=True)
    return out


def per_day(m, days):
    return {k: (v / days if k in ("value", "ceiling_value", "energy_payload_wh", "energy_total_wh", "isl_gb",
                                  "overflow", "expired", "processed", "offloaded", "tx_raw", "tx_prod",
                                  "generated", "delivered") else v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    from engine import RealAtlas
    atlas_days = RealAtlas.load(SimConfig()).days_done
    days = 3 if args.quick else atlas_days
    main_cfg = dict(seed=MAIN_SEED, duration_h=24.0 * days)
    day_cfg = dict(seed=MAIN_SEED, duration_h=24.0)
    V_grid = [0.03, 0.3, 3.0] if args.quick else [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    seeds = [1, 2] if args.quick else [1, 2, 3, 4, 5]

    jobs = []
    # 1. main long-horizon comparison + ablations
    for spec in [("elotus", {}), ("at", {}), ("gp", {}), ("sf", {}), ("rb", {}),
                 ("elotus", {"use_isl": False, "label": "E-LOTUS w/o ISLL"}),
                 ("elotus", {"use_entropy": False, "label": "E-LOTUS w/o entropy"}),
                 ("elotus", {"eclipse_aware": False, "label": "E-LOTUS static E_min"}),
                 ("elotus", {"V": 0.01, "eclipse_aware": False, "label": "E-LOTUS static E_min V=0.01"}),
                 ("elotus", {"estimator": "entropy", "label": "E-LOTUS raw entropy"})]:
        jobs.append(("main", main_cfg, spec, spec[0] in ("elotus", "gp") and not spec[1]))
    # 2. V sweep (V = 0.3 is the default and comes from the main run)
    for V in V_grid:
        if V != 0.3:
            jobs.append(("vsweep", main_cfg, ("elotus", {"V": V, "label": f"E-LOTUS V={V:g}"}), False))
    # 3. 24-h timeline trace (figure)
    for spec in [("elotus", {}), ("gp", {})]:
        jobs.append(("day", day_cfg, spec, True))
    # 4. multi-seed 24 h
    for s in seeds:
        for spec in [("elotus", {}), ("at", {}), ("gp", {}), ("sf", {}), ("rb", {})]:
            jobs.append(("seeds", dict(seed=s, start_day=4 * (s - 1), duration_h=24.0), spec, False))
    # 5. stress scenarios
    scen = {"load x1.5": {"load_scale": 1.5}, "radiator x0.75": {"k_thermal": 0.75},
            "SOC0 35%": {"soc0": 0.35}, "storage 4 GB": {"storage_mb": 4000.0},
            "Svalbard only": {"ground_stations": {"Svalbard (KSAT SvalSat)": (78.2297, 15.3975)}}}
    for nm, over in scen.items():
        for s in seeds[:3]:
            specs = [("elotus", {}), ("rb", {}), ("gp", {})]
            if nm == "SOC0 35%":
                specs.append(("elotus", {"eclipse_aware": False, "label": "E-LOTUS static E_min"}))
            for spec in specs:
                jobs.append((f"stress:{nm}", dict(seed=s, start_day=2 + 7 * (s - 1), duration_h=24.0, **over), spec, False))

    # group jobs by config so each worker reuses its ephemeris cache
    jobs.sort(key=lambda j: (str(sorted(j[1].items())), j[0]))
    t0 = time.time()
    print(f"running {len(jobs)} simulations on {args.procs} processes ...", flush=True)
    with Pool(args.procs) as pool:
        results = pool.map(_run, jobs, chunksize=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    summary = {"days": days, "main_seed": MAIN_SEED, "main": {}, "vsweep": {}, "seeds": {}, "stress": {}}
    for r in results:
        if r["tag"] == "main":
            summary["main"][r["policy"]] = per_day(r["metrics"], days)
            if r["policy"] == "E-LOTUS":
                summary["vsweep"]["0.3"] = per_day(r["metrics"], days)
        elif r["tag"] == "vsweep":
            summary["vsweep"][r["policy"].split("=")[1]] = per_day(r["metrics"], days)
        elif r["tag"] == "seeds":
            summary["seeds"].setdefault(r["policy"], []).append(r["metrics"])
        elif r["tag"].startswith("stress:"):
            summary["stress"].setdefault(r["tag"][7:], {}).setdefault(r["policy"], []).append(r["metrics"])

    def agg(lst):
        return {k: [float(np.mean([m[k] for m in lst])), float(np.std([m[k] for m in lst]))] for k in KEYS}

    summary["seeds"] = {p: agg(v) for p, v in summary["seeds"].items()}
    summary["stress"] = {s: {p: agg(v) for p, v in d.items()} for s, d in summary["stress"].items()}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # CSV exports
    def write_csv(path, rows, cols):
        with open(path, "w") as f:
            f.write(",".join(["name"] + cols) + "\n")
            for name, m in rows.items():
                f.write(",".join([name] + [f"{m[c]:.4f}" for c in cols]) + "\n")

    write_csv(os.path.join(RES, "main_30d.csv"), summary["main"], KEYS)
    write_csv(os.path.join(RES, "vsweep_30d.csv"), summary["vsweep"], KEYS)
    write_csv(os.path.join(RES, "multiseed_24h.csv"),
              {p: {k: v[k][0] for k in KEYS} for p, v in summary["seeds"].items()}, KEYS)
    with open(os.path.join(RES, "stress_24h.csv"), "w") as f:
        f.write("scenario,policy," + ",".join(KEYS) + "\n")
        for s, d in summary["stress"].items():
            for p, v in d.items():
                f.write(f"{s},{p}," + ",".join(f"{v[k][0]:.4f}" for k in KEYS) + "\n")

    make_figures(results, summary, days)
    print_tables(summary)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "axes.edgecolor": "#52514e",
                         "axes.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.dpi": 300,
                         "figure.dpi": 150})


def _shade_eclipse(ax, t, sunlit):
    ecl = np.asarray(sunlit) < 0.5
    edges = np.diff(np.r_[0, ecl.astype(int), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color="#1a1a19", alpha=0.07, lw=0)


def fig_telemetry(day, plt=None):
    """Fig. 3 from two 24-h telemetry dicts (E-LOTUS and Greedy-Process)."""
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _style(plt)
    cfg = SimConfig()
    el, gp = day["E-LOTUS"], day["Greedy-Process"]
    t = np.array(el["t_h"])
    fig, axs = plt.subplots(4, 1, figsize=(7.16, 4.6), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.32]})
    for ax in axs[:3]:
        _shade_eclipse(ax, t, el["sunlit"])
    axs[0].plot(t, gp["soc"], color=POLICY_COLORS["Greedy-Process"], label="Greedy-Process")
    axs[0].plot(t, el["soc"], color=POLICY_COLORS["E-LOTUS"], label="E-LOTUS")
    axs[0].plot(t, np.array(el["E_min_wh"]) / cfg.battery_wh * 100, color="#52514e", ls="--", lw=0.8,
                label=r"$E_{\min}(t)$ (eclipse-aware)")
    axs[0].axhline(cfg.e_crit_wh / cfg.battery_wh * 100, color="#e34948", lw=0.8, ls=":", label=r"$E_{\rm crit}$ (load-shed)")
    axs[0].set_ylabel("Battery SOC [%]")
    axs[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False)
    axs[1].plot(t, gp["temp_c"], color=POLICY_COLORS["Greedy-Process"])
    axs[1].plot(t, el["temp_c"], color=POLICY_COLORS["E-LOTUS"])
    axs[1].axhline(cfg.t_safe_c, color="#52514e", ls="--", lw=0.8)
    axs[1].axhline(cfg.t_max_c, color="#e34948", ls=":", lw=0.8)
    axs[1].text(t[-1], cfg.t_safe_c + 0.8, r"$T_{\rm safe}$", ha="right", fontsize=6.5, color="#52514e")
    axs[1].text(t[-1], cfg.t_max_c + 0.8, r"$T_{\max}$ (throttle)", ha="right", fontsize=6.5, color="#e34948")
    axs[1].set_ylabel(r"Node temp. [$^\circ$C]")
    axs[2].plot(t, np.array(gp["q_mb"]) / 1000, color=POLICY_COLORS["Greedy-Process"])
    axs[2].plot(t, np.array(el["q_mb"]) / 1000, color=POLICY_COLORS["E-LOTUS"])
    axs[2].set_ylabel("Backlog $q(t)$ [GB]")
    act = np.array(el["action"]).astype(int)
    for code, name in enumerate(ACTIONS):
        mask = act == code
        axs[3].bar(t[mask], np.ones(mask.sum()), width=t[1] - t[0], color=ACTION_COLORS[name], label=name, lw=0)
    axs[3].set_yticks([])
    axs[3].set_ylabel("E-LOTUS\naction", rotation=0, ha="right", va="center")
    axs[3].grid(False)
    axs[3].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.9), frameon=False)
    axs[3].set_xlabel("Mission elapsed time [h]  (shaded: eclipse)")
    axs[3].set_xlim(0, 24)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(FIG, "fig4_telemetry.png"), bbox_inches="tight")
    plt.close(fig)



def fig_realdata(plt=None):
    """Fig. 3: real NASA data along the PHISAT-2 track, entropy by class, held-out estimator evaluation."""
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _style(plt)
    from engine import RealAtlas
    from scipy.stats import spearmanr
    A = RealAtlas.load(SimConfig())
    cfg = SimConfig(duration_h=24.0 * A.days_done)
    eph = build_ephemeris(cfg)
    n = min(A.n, int(A.days_done * 86400 / A.step))
    step = int(A.step / cfg.dt)
    sun = eph.sunlit[:n * step:step][:n]
    H = np.where(sun, A.H_tc[:n], A.H_bt[:n]) / 30.0
    cl = np.where(sun, A.cloud_day[:n], A.cloud_night[:n]).astype(float)
    cl_alt = np.where(sun, A.cloud_night[:n], A.cloud_day[:n]).astype(float)
    cl = np.where(cl == 255, cl_alt, cl)
    cloud = np.where(cl == 255, 50.0, cl) / 100.0
    land = A.land[:n] / 100.0
    alert = A.fire[:n] >= cfg.fire_trigger_px
    dark = sun & (A.mean_tc[:n] < 8)
    val = np.where(alert, 3.0, np.where(dark, 0.02, 0.02 + 0.63 * land * (1 - cloud)))
    lab = np.array(["fire" if a else "dark" if dk else "cloud" if c >= 0.6 else "ocean" if l < 0.1 else "coast" if l < 0.6 else "land"
                    for a, dk, c, l in zip(alert, dark, cloud, land)])
    # held-out evaluation of value estimators (second half of the window), alerts excluded (flagged by trigger)
    half = n // 2
    held = (np.arange(n) >= half) & ~alert
    g = 0.1 + 0.9 * land
    ests = {"random": np.random.default_rng(0).random(n), "raw entropy": (H / 8) ** 3, "land prior": g,
            "E-LOTUS": np.where(sun, (H / 8) ** 3 * g, g), "oracle": val}

    def capture(score, mask, frac=0.3):
        s_ = score[mask] + 1e-9 * np.random.default_rng(1).random(mask.sum())
        v_ = val[mask]
        idx = np.argsort(-s_)[:int(frac * mask.sum())]
        return float(v_[idx].sum() / v_.sum())

    cap = {k: capture(v, held) for k, v in ests.items()}
    rho = {k: float(spearmanr(v[held], val[held]).correlation) for k, v in ests.items() if k != "random"}
    within = {"optical": float(spearmanr(H[sun & ~alert & ~dark & (land > 0.8)], 1 - cloud[sun & ~alert & ~dark & (land > 0.8)]).correlation),
              "tir": float(spearmanr(H[~sun & ~alert & (land > 0.8)], 1 - cloud[~sun & ~alert & (land > 0.8)]).correlation)}
    classes = ["dark", "ocean", "cloud", "coast", "land", "fire"]
    stats = {"n": int(n), "days": int(A.days_done), "dates": [A.header["dates"][0], A.header["dates"][A.days_done - 1]],
             "alert_rate_pct": float(100 * alert.mean()),
             "rho_raw_optical": float(spearmanr(H[sun & ~alert], val[sun & ~alert]).correlation),
             "rho_raw_tir": float(spearmanr(H[~sun & ~alert], val[~sun & ~alert]).correlation),
             "within_land_rho_H_clear": within, "heldout_capture_top30": cap, "heldout_rho": rho,
             "H_by_class_optical": {c: [float(np.median(H[sun & (lab == c)])) if (sun & (lab == c)).any() else float("nan"),
                                        int((sun & (lab == c)).sum())] for c in classes},
             "share_by_class_pct": {c: float(100 * (lab == c).mean()) for c in classes}}
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [1.75, 1.0, 0.9]})
    d0 = slice(0, int(86400 / A.step))
    sc = axs[0].scatter(A.lon[d0], A.lat[d0], c=H[d0], s=1.3, cmap="viridis", vmin=0, vmax=8, lw=0, rasterized=True)
    fa = alert[d0]
    axs[0].scatter(A.lon[d0][fa], A.lat[d0][fa], s=9, facecolor="none", edgecolor="#e34948", lw=0.7, label="VIIRS fire trigger")
    for nm, (la, lo) in cfg.ground_stations.items():
        axs[0].scatter([lo], [la], marker="^", s=22, color="#0b0b0b", zorder=4)
        axs[0].annotate(nm.split(" (")[0], (lo, la), xytext=(3, 3), textcoords="offset points", fontsize=5.8)
    axs[0].set_xlim(-180, 180); axs[0].set_ylim(-90, 90)
    axs[0].set_xlabel("Longitude [deg]"); axs[0].set_ylabel("Latitude [deg]")
    axs[0].set_title(f"(a) PHISAT-2 track {A.header['dates'][0]}, quick-look $H$")
    cb = fig.colorbar(sc, ax=axs[0], pad=0.01, fraction=0.045); cb.set_label("$H$ [bit/px]")
    axs[0].legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=5.6)
    data = [H[sun & (lab == c)] for c in classes]
    bp = axs[1].boxplot(data, positions=range(len(classes)), widths=0.55, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cde2fb"); patch.set_edgecolor("#2a78d6")
    for med in bp["medians"]:
        med.set_color("#0b0b0b")
    axs[1].set_xticks(range(len(classes)))
    axs[1].set_xticklabels([f"{c}\n{stats['share_by_class_pct'][c]:.0f}%" for c in classes], fontsize=5.4, rotation=40)
    axs[1].set_ylabel("Optical $H$ [bit/px]")
    axs[1].set_title("(b) Real entropy by ground truth")
    names = ["random", "raw entropy", "land prior", "E-LOTUS", "oracle"]
    cols = ["#c9c8c1", "#eb6834", "#9b9a94", "#2a78d6", "#52514e"]
    axs[2].barh(range(len(names)), [100 * cap[k] for k in names], color=cols, height=0.62)
    for i, k in enumerate(names):
        axs[2].text(100 * cap[k] + 1.5, i, f"{100 * cap[k]:.0f}%", va="center", fontsize=6)
    axs[2].set_yticks(range(len(names))); axs[2].set_yticklabels(names, fontsize=6.2)
    axs[2].invert_yaxis(); axs[2].set_xlim(0, 100); axs[2].grid(axis="y", visible=False)
    axs[2].set_xlabel("Value in top-30% ranked [%]")
    axs[2].set_title("(c) Held-out triage")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, "fig3_realdata.png"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(RES, "realdata_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def fig_pareto(summary, plt=None):
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _style(plt)
    # --- Fig. 4: Pareto frontier (value/day vs payload energy/day) ---
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    vs = sorted(summary["vsweep"].items(), key=lambda kv: float(kv[0]))
    xs = [m["energy_payload_wh"] for _, m in vs]
    ys = [m["value"] for _, m in vs]
    ax.plot(xs, ys, color=POLICY_COLORS["E-LOTUS"], marker="o", ms=4, label="E-LOTUS (sweep of $V$)")
    ax.annotate(f"E-LOTUS, V = {float(vs[0][0]):g} ... {float(vs[-1][0]):g}", (xs[-1], max(ys)), textcoords="offset points",
                xytext=(-8, 8), ha="right", fontsize=6.2, color=POLICY_COLORS["E-LOTUS"])
    for p in ["Always-Transmit", "Greedy-Process", "Store-and-Forward", "Rule-Based"]:
        m = summary["main"][p]
        ax.annotate(p, (m["energy_payload_wh"], m["value"]), textcoords="offset points",
                    xytext=(-6, -10) if p == "Rule-Based" else (6, 3), ha="right" if p == "Rule-Based" else "left",
                    fontsize=6.2, color="#52514e")
    markers = {"Always-Transmit": "s", "Greedy-Process": "^", "Store-and-Forward": "D", "Rule-Based": "P"}
    for p, mk in markers.items():
        m = summary["main"][p]
        ax.scatter(m["energy_payload_wh"], m["value"], marker=mk, s=28, color=POLICY_COLORS[p],
                   edgecolor="white", linewidth=0.8, zorder=3, label=p)
    ax.set_xlabel("Payload energy [Wh/day]")
    ax.set_ylabel("Semantic value delivered [/day]")
    ax.set_xlim(30, 240)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_pareto.png"), bbox_inches="tight")
    plt.close(fig)



def make_figures(results, summary, days):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    day = {r["policy"]: r["tel"] for r in results if r["tag"] == "day"}
    main = {r["policy"]: r for r in results if r["tag"] == "main"}
    cfg = SimConfig()

    fig_telemetry(day, plt)
    fig_realdata(plt)

    fig_pareto(summary, plt)
    vs = sorted(summary["vsweep"].items(), key=lambda kv: float(kv[0]))

    # --- Fig. 5: Lyapunov trade-off parameter sweep ---
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.75))
    Vv = [float(V) for V, _ in vs]
    panels = [([m["value"] - 0.5 * m["energy_payload_wh"] for _, m in vs], r"Net utility $\bar U-\zeta\bar{\mathcal{E}}$ [/day]", POLICY_COLORS["E-LOTUS"]),
              ([m["backlog_mean_mb"] / 1000 for _, m in vs], r"Mean backlog $\bar q$ [GB]", "#52514e"),
              ([m["soc_min_pct"] for _, m in vs], r"Minimum SOC [%]", "#52514e")]
    for ax, (y, title, col) in zip(axs, panels):
        ax.plot(Vv, y, marker="o", ms=3.5, color=col)
        ax.set_xscale("log")
        ax.set_xlabel("$V$")
        ax.set_title(title)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIG, "fig6_vsweep.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Fig. 6: cumulative value over the long horizon ---
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for p in ["E-LOTUS", "Greedy-Process"]:
        tel = main[p]["tel"]
        th = np.array(tel["t_h"]) / 24.0
        ax.plot(th[::60], np.array(tel["value_cum"])[::60], color=POLICY_COLORS[p], label=p)
    ax.set_xlabel("Mission day")
    ax.set_ylabel("Cumulative semantic value")
    for p in ["Rule-Based", "Store-and-Forward", "Always-Transmit"]:
        m = summary["main"][p]
        ax.scatter([days], [m["value"] * days], color=POLICY_COLORS[p], s=18, zorder=3, label=f"{p} (day {days})",
                   edgecolor="white", linewidth=0.6)
    ax.legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_cumulative.png"), bbox_inches="tight")
    plt.close(fig)


def print_tables(summary):
    print("\n=== MAIN (per day averages over %d days, seed %d) ===" % (summary["days"], summary["main_seed"]))
    cols = ["value", "ceiling_pct", "completion_pct", "alerts_on_time_pct", "energy_payload_wh", "value_per_wh",
            "latency_mean_min", "soc_min_pct", "temp_max_c", "shed_pct", "throttled_pct", "gpu_duty_pct",
            "dl_util_pct", "overflow", "offloaded"]
    print(f"{'policy':24s}" + "".join(f"{c[:10]:>11s}" for c in cols))
    for p, m in summary["main"].items():
        print(f"{p:24s}" + "".join(f"{m[c]:11.2f}" for c in cols))
    print("\n=== V SWEEP ===")
    for V, m in sorted(summary["vsweep"].items(), key=lambda kv: float(kv[0])):
        print(f"V={V:6s}" + "".join(f"{m[c]:11.2f}" for c in cols) + f"  q={m['backlog_mean_mb']:.0f}")
    print("\n=== MULTI-SEED 24 h (mean ± std) ===")
    for p, d in summary["seeds"].items():
        print(f"{p:24s}" + "".join(f"{d[c][0]:8.1f}±{d[c][1]:<5.1f}" for c in cols[:8]))
    print("\n=== STRESS ===")
    for s, dd in summary["stress"].items():
        for p, d in dd.items():
            print(f"{s:16s}{p:18s}" + "".join(f"{d[c][0]:8.1f}±{d[c][1]:<5.1f}" for c in cols[:9]))


if __name__ == "__main__":
    main()
