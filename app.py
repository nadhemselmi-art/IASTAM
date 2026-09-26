"""
Orbitra Mission Control -- Streamlit dashboard (powered by the E-LOTUS decision engine)
Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import datetime as _dt

from engine import (ACTIONS, GROUND_STATIONS, KELVIN, R_EARTH, AlwaysTransmit, ELotusEngine, GreedyProcess,
                    RuleBased, SimConfig, Simulation, StoreAndForward, build_ephemeris, generate_workload,
                    gmst_rad, orbit_power_map)

from engine import RealAtlas
_ATLAS = RealAtlas.load(SimConfig())
REAL_DATES = list(_ATLAS.header["dates"][:_ATLAS.days_done])

st.set_page_config(page_title="Orbitra Mission Control", page_icon="🛰️", layout="wide")

POLICY_COLORS = {"E-LOTUS": "#2a78d6", "Always-Transmit": "#eb6834", "Greedy-Process": "#1baf7a",
                 "Store-and-Forward": "#eda100", "Rule-Based": "#e87ba4"}
ACTION_COLORS = {"STORE": "#9b9a94", "PROCESS": "#4a3aa7", "TRANSMIT": "#eb6834", "OFFLOAD_ISLL": "#1baf7a"}
SUN_COLOR, ECL_COLOR = "#eda100", "#4a3aa7"
BASELINES = ["Always-Transmit", "Greedy-Process", "Store-and-Forward", "Rule-Based"]
V_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------
def cfg_from(params: dict) -> SimConfig:
    p = dict(params)
    gs = p.pop("stations")
    return SimConfig(ground_stations={k: GROUND_STATIONS[k] for k in gs}, **p)


@st.cache_resource(show_spinner=False)
def world(params_json: str):
    cfg = cfg_from(json.loads(params_json))
    eph = build_ephemeris(cfg)
    return cfg, eph, generate_workload(cfg, eph)


def _policy(name: str, V: float, zeta: float):
    if name == "E-LOTUS":
        return ELotusEngine(V=V, zeta=zeta)
    if name.startswith("E-LOTUS V="):
        return ELotusEngine(V=float(name.split("=")[1]), zeta=zeta, label=name)
    return {"Always-Transmit": AlwaysTransmit, "Greedy-Process": GreedyProcess,
            "Store-and-Forward": StoreAndForward, "Rule-Based": RuleBased}[name]()


@st.cache_data(show_spinner=False, max_entries=64)
def simulate(params_json: str, name: str, V: float, zeta: float):
    cfg, eph, arr = world(params_json)
    res = Simulation(cfg, _policy(name, V, zeta), eph, arr).run()
    return {"telemetry": res.telemetry, "events": res.events, "metrics": res.metrics}


@st.cache_resource(show_spinner=False)
def real_orbit_info():
    """Mean altitude, inclination and ascending-node local time of the real tracked orbit."""
    c = SimConfig()
    e = build_ephemeris(c, extra_h=0.0)
    h = np.cross(e.r[0], e.v[0])
    node = np.cross([0.0, 0.0, 1.0], h)
    raan = math.degrees(math.atan2(node[1], node[0]))
    ra_sun = math.degrees(math.atan2(e.sun[0, 1], e.sun[0, 0]))
    return dict(alt=round(float(e.alt.mean())), inc=round(math.degrees(math.acos(h[2] / np.linalg.norm(h))), 2),
                ltan=((raan - ra_sun) / 15.0 + 12.0) % 24.0)


def fmt_lt(h: float) -> str:
    h = h % 24.0
    return f"{int(h):02d}:{int(round(h % 1 * 60)) % 60:02d}"


@st.cache_resource(show_spinner=False)
def earth_texture(step: float = 1.5):
    """NASA Blue Marble (public domain) sampled on a lat/lon grid + triangle indices of the sphere mesh."""
    from PIL import Image
    nlon, nlat = int(360 / step) + 1, int(180 / step) + 1
    im = Image.open(os.path.join(HERE, "data", "earth_texture.jpg")).convert("RGB").resize((nlon, nlat), Image.LANCZOS)
    rgb = np.asarray(im, float)
    LO, LA = np.meshgrid(np.radians(np.linspace(-180, 180, nlon)), np.radians(np.linspace(90, -90, nlat)))
    ii = np.arange(nlat - 1)[:, None] * nlon + np.arange(nlon - 1)[None, :]
    a = ii.ravel()
    tri = (np.concatenate([a, a + 1]), np.concatenate([a + nlon, a + nlon]), np.concatenate([a + 1, a + nlon + 1]))
    return rgb, LO, LA, tri


def earth_surface(jd: float, sun: np.ndarray, radius: float = R_EARTH) -> go.Mesh3d:
    """Textured Earth in the inertial frame: rotated by GMST, night side darkened along the real terminator."""
    rgb, LO, LA, (I, J, K) = earth_texture()
    g = float(gmst_rad(np.array([jd]))[0])
    ux, uy, uz = np.cos(LA) * np.cos(LO + g), np.cos(LA) * np.sin(LO + g), np.sin(LA)
    c = ux * sun[0] + uy * sun[1] + uz * sun[2]
    f = (0.16 + 0.84 * np.clip((c + 0.08) / 0.16, 0, 1) ** 1.5)[..., None]
    col = np.clip(rgb * f + (1 - f) * np.array([1.0, 4.0, 11.0]), 0, 255).astype(int).reshape(-1, 3)
    return go.Mesh3d(x=(radius * ux).ravel(), y=(radius * uy).ravel(), z=(radius * uz).ravel(), i=I, j=J, k=K,
                     vertexcolor=["#%02x%02x%02x" % tuple(v) for v in col], name="Earth", hoverinfo="skip",
                     flatshading=False, lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0))


INC_MAP = np.arange(40.0, 100.01, 2.5)
LTAN_MAP = np.arange(0.0, 24.0, 0.5)
INC_OPT = [45.0, 60.0, 75.0, 90.0, 97.4]
LTAN_OPT = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]


@st.cache_data(show_spinner=False)
def power_map(start_day: int, stations: tuple, alt: float):
    c = SimConfig(start_day=start_day, altitude_km=alt,
                  ground_stations={k: GROUND_STATIONS[k] for k in stations})
    return orbit_power_map(c, INC_MAP, LTAN_MAP)


@st.cache_data(show_spinner=False, max_entries=512)
def orbit_point(base_json: str, inc: float, lt: float, V: float, zeta: float, alt: float):
    """One full E-LOTUS run on a what-if orbit (same settings as the sidebar)."""
    p = dict(json.loads(base_json), orbit_model="rk4_j2", inclination_deg=inc, ltan_h=lt, altitude_km=alt)
    c = cfg_from(p)
    e = build_ephemeris(c)
    r = Simulation(c, ELotusEngine(V=V, zeta=zeta), e, generate_workload(c, e)).run()
    return {k: float(r.metrics[k]) for k in ("value", "soc_min_pct", "shed_pct")}


def optimise_orbit(base_json: str, V: float, zeta: float, alt: float):
    out = {k: np.zeros((len(INC_OPT), len(LTAN_OPT))) for k in ("value", "soc_min_pct", "shed_pct")}
    bar = st.progress(0.0, text="Running the decision engine on every candidate orbit ...")
    n = len(INC_OPT) * len(LTAN_OPT)
    for i, inc in enumerate(INC_OPT):
        for j, lt in enumerate(LTAN_OPT):
            m_ = orbit_point(base_json, inc, lt, V, zeta, alt)
            for k in out:
                out[k][i, j] = m_[k]
            done = i * len(LTAN_OPT) + j + 1
            bar.progress(done / n, text=f"Orbit {done}/{n}: inclination {inc:g}°, ascending node {fmt_lt(lt)}")
    bar.empty()
    return out


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div style='font-weight:800;font-size:28px;letter-spacing:.14em;line-height:1'>ORBITRA</div>"
                    "<div style='opacity:.65;font-size:13px;margin:4px 0 2px'>Onboard intelligence for Earth observation"
                    "</div>", unsafe_allow_html=True)
st.sidebar.caption("Powered by the E-LOTUS decision engine")
st.sidebar.subheader("What-if simulation")
V = st.sidebar.select_slider("Lyapunov trade-off parameter V", options=V_GRID, value=0.3,
                             help="Large V favours semantic utility; small V favours queue, energy and thermal stability. "
                                  "Theory: utility gap O(1/V), backlog O(V).")
soc0 = st.sidebar.slider("Initial battery SOC [%]", 25, 100, 70, 1)
storage_gb = st.sidebar.slider("Storage limit [GB]", 2.0, 32.0, 8.0, 0.5)
k_th = st.sidebar.slider("Thermal dissipation coefficient k", 0.50, 1.50, 1.00, 0.05,
                         help="Scales the radiator conductance: Q_rad = k * eps * sigma * A * (T^4 - T_space^4)")
isl = st.sidebar.toggle("ISLL offload window (swarm)", value=True,
                        help="Allow offloading to the leading/trailing neighbour while it is sunlit and this node is in eclipse.")
day_label = st.sidebar.select_slider("Real day (NASA VIIRS / FIRMS data)", options=REAL_DATES, value=REAL_DATES[0],
                                     help="24-h window along the real ground track; imagery, fires and clouds of that date.")
start_day = REAL_DATES.index(day_label)

REAL = real_orbit_info()
st.session_state.setdefault("orbit_choice", "Real tracked orbit")
st.session_state.setdefault("inc", round(float(REAL["inc"]), 1))
st.session_state.setdefault("ltan", round(REAL["ltan"] * 2) / 2 % 24)
st.sidebar.subheader("Orbit designer")
orbit_choice = st.sidebar.radio("Orbit", ["Real tracked orbit", "What-if orbit"], horizontal=True, key="orbit_choice",
                                help="What-if orbits are integrated with RK4 two-body + J2 at the real mean altitude; "
                                     "each footprint uses the nearest real NASA scene.")
whatif = orbit_choice == "What-if orbit"
inc = st.sidebar.slider("Inclination [°]", 40.0, 100.0, step=0.1, key="inc", disabled=not whatif,
                        help="Tilt of the orbit plane. Sets which latitudes you overfly and how often the "
                             "polar ground station (Svalbard, 78° N) is in view.")
ltan = st.sidebar.select_slider("Ascending-node local time", options=[x / 2 for x in range(48)],
                                format_func=fmt_lt, key="ltan",
                                disabled=not whatif,
                                help="Where the orbit plane sits relative to the Sun. 06:00 / 18:00 = dawn-dusk "
                                     "(almost no eclipse, Sun edge-on to the body panels); 00:00 / 12:00 = noon-midnight "
                                     "(longest eclipse, Sun overhead at noon).")
st.sidebar.caption(f"Real orbit: {REAL['inc']:.1f}° · node {fmt_lt(REAL['ltan'])} · {REAL['alt']} km")
with st.sidebar.expander("Advanced"):
    max_h = int(min(48, (len(REAL_DATES) - start_day) * 24))
    horizon = st.slider("Horizon [h]", 6, max_h, min(24, max_h), 6)
    workload = st.radio("Workload", ["real", "synthetic"], index=0, horizontal=True,
                        help="real = NASA GIBS imagery along the real track; synthetic = procedural thumbnails")
    load = st.slider("Instrument load scale", 0.5, 2.0, 1.0, 0.1)
    zeta = st.slider("Energy price zeta [value/Wh]", 0.0, 2.0, 0.5, 0.1)
    seed = st.number_input("Workload seed", 0, 9999, 2026, 1)
    stations = st.multiselect("Ground stations", list(GROUND_STATIONS), default=list(GROUND_STATIONS))
    if not stations:
        stations = [list(GROUND_STATIONS)[0]]
compare_with = st.sidebar.multiselect("Overlay baselines in telemetry", BASELINES, default=["Greedy-Process"])

params = dict(soc0=soc0 / 100.0, storage_mb=storage_gb * 1000.0, k_thermal=k_th, isl_enabled=isl,
              duration_h=float(horizon), load_scale=load, seed=int(seed), stations=stations,
              start_day=start_day, workload=workload)
BASE_PJ = json.dumps(params, sort_keys=True)          # orbit-independent settings
if whatif:
    params.update(orbit_model="rk4_j2", inclination_deg=float(inc), ltan_h=float(ltan), altitude_km=float(REAL["alt"]))
PJ = json.dumps(params, sort_keys=True)
cfg, eph, arrivals = world(PJ)
N = cfg.n_slots

with st.spinner("Running the E-LOTUS engine and the baselines over the orbit ..."):
    runs = {"E-LOTUS": simulate(PJ, "E-LOTUS", V, zeta)}
    for b in BASELINES:
        runs[b] = simulate(PJ, b, V, zeta)
el = runs["E-LOTUS"]
tel = el["telemetry"]

# ---------------------------------------------------------------------------
# Header and KPI tiles
# ---------------------------------------------------------------------------
st.markdown("<div style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap'>"
            "<span style='font-weight:800;font-size:40px;letter-spacing:.12em'>ORBITRA</span>"
            "<span style='font-size:20px;opacity:.7'>Mission Control</span></div>", unsafe_allow_html=True)
orbit_txt = (f"What-if orbit: inclination {inc:.1f}°, ascending node {fmt_lt(ltan)} local time, {REAL['alt']} km "
             f"(RK4 two-body + J2, nearest real NASA scenes)" if whatif
             else "Real tracked orbit (public TLE, SGP4)")
st.caption(f"{orbit_txt} | altitude {eph.alt[:N].mean():.0f} km | beta {eph.beta_deg[0]:.1f}° | "
           f"sunlit {eph.sunlit[:N].mean() * 100:.0f}% | ground contact {eph.contact[:N].sum() * cfg.dt / 60:.0f} min over {horizon} h | "
           f"{'NASA GIBS VIIRS/MODIS data for ' + day_label if workload == 'real' else 'synthetic workload'} | "
           f"tasks {el['metrics']['generated']}")
if el["metrics"]["shed_pct"] > 5:
    st.warning(f"This orbit cannot power the payload: the battery hits the load-shed floor "
               f"{el['metrics']['shed_pct']:.0f}% of the time. Try another plane in the Orbit designer.")

m, rb = el["metrics"], runs["Rule-Based"]["metrics"]
c = st.columns(6)
c[0].metric("Semantic value delivered", f"{m['value']:.1f}", f"{(m['value'] / max(rb['value'], 1e-9) - 1) * 100:+.1f}% vs Rule-Based")
c[1].metric("Share of contact-limited ceiling", f"{m['ceiling_pct']:.1f}%", f"{m['ceiling_pct'] - rb['ceiling_pct']:+.1f} pts")
c[2].metric("Task completion", f"{m['completion_pct']:.1f}%", f"{m['completion_pct'] - rb['completion_pct']:+.1f} pts")
c[3].metric("Payload energy", f"{m['energy_payload_wh']:.0f} Wh", f"{m['energy_payload_wh'] - rb['energy_payload_wh']:+.0f} Wh", delta_color="inverse")
c[4].metric("Minimum battery SOC", f"{m['soc_min_pct']:.0f}%", f"max temp. {m['temp_max_c']:.0f} °C", delta_color="off")
c[5].metric("Mean delivery latency", f"{m['latency_mean_min']:.0f} min", f"{m['latency_mean_min'] - rb['latency_mean_min']:+.0f} min", delta_color="inverse")

tabs = st.tabs(["Orbit", "Subsystem telemetry", "Pareto frontier", "Live decision feed", "Benchmark table", "Model"])


# ---------------------------------------------------------------------------
# Orbit tab
# ---------------------------------------------------------------------------
def footprint(lat0, lon0, lam_deg, n=90):
    lat0, lon0, lam = map(math.radians, (lat0, lon0, lam_deg))
    br = np.linspace(0, 2 * np.pi, n)
    lat = np.arcsin(np.sin(lat0) * np.cos(lam) + np.cos(lat0) * np.sin(lam) * np.cos(br))
    lon = lon0 + np.arctan2(np.sin(br) * np.sin(lam) * np.cos(lat0), np.cos(lam) - np.sin(lat0) * np.sin(lat))
    return np.degrees(lat), (np.degrees(lon) + 180) % 360 - 180


def split_wrap(lon, lat, mask):
    lo, la = lon.astype(float).copy(), lat.astype(float).copy()
    lo[~mask] = np.nan
    la[~mask] = np.nan
    jump = np.abs(np.diff(lon)) > 180
    lo[1:][jump] = np.nan
    la[1:][jump] = np.nan
    return lo, la


with tabs[0]:
    t_scrub = st.slider("Mission time [h]", 0.0, float(horizon) - cfg.dt / 3600, 1.0, 1 / 60, key="scrub")
    k = min(int(t_scrub * 3600 / cfg.dt), N - 1)
    s = st.columns(5)
    s[0].metric("State", "SUNLIT" if eph.sunlit[k] else "ECLIPSE")
    s[1].metric("Battery", f"{tel['soc'][k]:.1f}%")
    s[2].metric("Node temperature", f"{tel['temp_c'][k]:.1f} °C")
    s[3].metric("Backlog q(t)", f"{tel['q_mb'][k] / 1000:.2f} GB")
    s[4].metric("E-LOTUS action", ACTIONS[int(tel['action'][k])])
    col_a, col_b = st.columns(2)
    with col_a:
        lat, lon = eph.lat[:N], eph.lon[:N]
        fig = go.Figure()
        lam = 90 - cfg.elevation_mask_deg - math.degrees(math.asin(R_EARTH * math.cos(math.radians(cfg.elevation_mask_deg))
                                                                    / (R_EARTH + cfg.altitude_km)))
        for nm in cfg.ground_stations:
            glat, glon = cfg.ground_stations[nm]
            flat, flon = footprint(glat, glon, lam)
            fig.add_trace(go.Scattergeo(lat=flat, lon=flon, mode="lines", line=dict(color="#52514e", width=1, dash="dot"),
                                        name=f"{nm} pass zone", hoverinfo="skip"))
            fig.add_trace(go.Scattergeo(lat=[glat], lon=[glon], mode="markers+text", text=[nm], textposition="bottom center",
                                        marker=dict(symbol="triangle-up", size=10, color="#0b0b0b"), showlegend=False))
        for label, mask, color in (("Sunlit", eph.sunlit[:N], SUN_COLOR), ("Eclipse", ~eph.sunlit[:N], ECL_COLOR)):
            lo, la = split_wrap(lon, lat, mask)
            fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", line=dict(color=color, width=1.6), name=label))
        lo, la = split_wrap(lon, lat, eph.contact[:N])
        fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", line=dict(color="#e34948", width=4), name="Ground contact"))
        tasks_all = [tk for slot in arrivals[:N] for tk in slot]
        if tasks_all and workload == "real":
            fig.add_trace(go.Scattergeo(lat=[tk.lat for tk in tasks_all], lon=[tk.lon for tk in tasks_all], mode="markers",
                                        name="Real scenes (colour = entropy H)",
                                        marker=dict(size=4, color=[tk.entropy for tk in tasks_all], colorscale="Viridis",
                                                    cmin=0, cmax=8, colorbar=dict(title="H", len=0.5, thickness=10)),
                                        text=[f"{tk.scene} | H={tk.entropy:.2f} | land {tk.land:.0%} | cloud {tk.cloud:.0%} | fire px {tk.fire_px}"
                                              for tk in tasks_all], hoverinfo="text"))
            al = [tk for tk in tasks_all if tk.priority == 4]
            fig.add_trace(go.Scattergeo(lat=[tk.lat for tk in al], lon=[tk.lon for tk in al], mode="markers",
                                        name="VIIRS fire alerts", marker=dict(size=9, symbol="circle-open", color="#e34948",
                                                                              line=dict(width=2))))
        fig.add_trace(go.Scattergeo(lat=[lat[k]], lon=[lon[k]], mode="markers", name="Node",
                                    marker=dict(size=13, color="#2a78d6", line=dict(color="white", width=2))))
        fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#d8d7d0",
                        showland=True, landcolor="#f0efec", showocean=True, oceancolor="#e7eef7", coastlinecolor="#b8b7b0")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=30, b=0), title="Ground track (2D)",
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
    with col_b:
        fig3 = go.Figure()
        jd_k = cfg.epoch_jd + k * cfg.dt / 86400.0
        fig3.add_trace(earth_surface(jd_k, eph.sun[k]))
        g_k = float(gmst_rad(np.array([jd_k]))[0])
        for nm, (glat, glon) in cfg.ground_stations.items():
            la_, lo_ = math.radians(glat), math.radians(glon) + g_k
            gp = (R_EARTH + 40) * np.array([math.cos(la_) * math.cos(lo_), math.cos(la_) * math.sin(lo_), math.sin(la_)])
            fig3.add_trace(go.Scatter3d(x=[gp[0]], y=[gp[1]], z=[gp[2]], mode="markers+text", text=[nm.split(" (")[0]],
                                        textfont=dict(color="#ffffff", size=11), showlegend=False, hoverinfo="text",
                                        marker=dict(size=5, symbol="diamond", color="#3ddc97")))
        span = slice(max(0, k - 300), min(N, k + 300))
        r = eph.r[span]
        sl = eph.sunlit[span]
        for label, mask, color in (("Sunlit", sl, SUN_COLOR), ("Eclipse", ~sl, ECL_COLOR)):
            rr = r.copy()
            rr[~mask] = np.nan
            fig3.add_trace(go.Scatter3d(x=rr[:, 0], y=rr[:, 1], z=rr[:, 2], mode="lines", line=dict(color=color, width=5), name=label))
        rk = eph.r[k]
        hh = np.cross(eph.r[k], eph.v[k])
        hh /= np.linalg.norm(hh)
        phi = math.radians(cfg.isl_phase_deg)
        for side, sg in (("lead", 1), ("trail", -1)):
            rn = rk * math.cos(phi) + sg * np.cross(hh, rk) * math.sin(phi)
            ok = bool(eph.isl_los[side][k] and eph.nb_sunlit[side][k])
            fig3.add_trace(go.Scatter3d(x=[rk[0], rn[0]], y=[rk[1], rn[1]], z=[rk[2], rn[2]], mode="lines+markers",
                                        line=dict(color="#1baf7a" if ok else "#c9c8c1", width=4, dash="solid" if ok else "dot"),
                                        marker=dict(size=[0, 5], color="#52514e"),
                                        name=f"ISLL {side} ({'available' if ok else 'neighbour in shadow'})"))
        fig3.add_trace(go.Scatter3d(x=[rk[0]], y=[rk[1]], z=[rk[2]], mode="markers", name="Node",
                                    marker=dict(size=7, color="#2a78d6")))
        sd = eph.sun[k] * (R_EARTH + 2500)
        fig3.add_trace(go.Scatter3d(x=[0, sd[0]], y=[0, sd[1]], z=[0, sd[2]], mode="lines", name="Sun direction",
                                    line=dict(color=SUN_COLOR, width=3, dash="dash")))
        lim = R_EARTH + 1200
        fig3.update_layout(height=430, margin=dict(l=0, r=0, t=30, b=0), title="Live orbit (3D, inertial frame, +/- 50 min)",
                           paper_bgcolor="#060a14", font=dict(color="#e6ebf5"),
                           scene=dict(xaxis=dict(visible=False, range=[-lim, lim]), yaxis=dict(visible=False, range=[-lim, lim]),
                                      zaxis=dict(visible=False, range=[-lim, lim]), aspectmode="cube", bgcolor="#060a14",
                                      camera=dict(eye=dict(x=1.35 * float(eph.r[k][0]) / np.linalg.norm(eph.r[k]) + 0.25,
                                                           y=1.35 * float(eph.r[k][1]) / np.linalg.norm(eph.r[k]) + 0.25,
                                                           z=1.35 * float(eph.r[k][2]) / np.linalg.norm(eph.r[k]) + 0.2))),
                           legend=dict(orientation="h", y=-0.05, font=dict(color="#e6ebf5")),
                           title_font=dict(color="#e6ebf5"))
        st.plotly_chart(fig3, width="stretch", theme=None)
    st.caption(f"Solar incidence on zenith panel theta(t) = {eph.theta_deg[k]:.1f} deg | array power {tel['p_solar_w'][k]:.1f} W | "
               f"load {tel['p_load_w'][k]:.1f} W | eclipse-aware floor E_min(t) = {tel['E_min_wh'][k]:.1f} Wh")

    # ---------------- Orbit designer ----------------
    st.markdown("### Orbit designer: which orbit keeps the battery healthiest?")
    st.caption("Each cell is one candidate orbit plane at the real altitude. The white ring is the orbit you are "
               "simulating; the cross is the real tracked orbit. Change the orbit in the sidebar (Orbit designer).")
    pm = power_map(start_day, tuple(stations), float(REAL["alt"]))
    cur_inc = float(inc) if whatif else REAL["inc"]
    cur_lt = float(ltan) if whatif else REAL["ltan"]

    def orbit_heatmap(z, title, unit, scale, x=LTAN_MAP, y=INC_MAP, fmt=".0f"):
        f = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=scale, colorbar=dict(title=unit, thickness=10),
                                 hovertemplate="node %{x:.1f} h<br>inclination %{y:.1f}°<br>%{z:" + fmt + "} " + unit
                                               + "<extra></extra>"))
        f.add_trace(go.Scatter(x=[cur_lt], y=[cur_inc], mode="markers", showlegend=False, hoverinfo="skip",
                               marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="white", width=3))))
        f.add_trace(go.Scatter(x=[REAL["ltan"]], y=[REAL["inc"]], mode="markers", showlegend=False, hoverinfo="skip",
                               marker=dict(size=11, symbol="x-thin", line=dict(color="#e34948", width=3))))
        f.update_layout(height=300, title=title, margin=dict(l=10, r=10, t=40, b=10),
                        xaxis=dict(title="Ascending-node local time [h]", tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
                                   ticktext=["00", "03", "06", "09", "12", "15", "18", "21"]),
                        yaxis=dict(title="Inclination [°]"))
        return f

    d1, d2, d3 = st.columns(3)
    d1.plotly_chart(orbit_heatmap(pm["solar_wh"], "Solar energy available per day", "Wh", "Viridis"), width="stretch")
    d2.plotly_chart(orbit_heatmap(pm["eclipse_pct"], "Time in Earth's shadow", "%", "Magma_r"), width="stretch")
    d3.plotly_chart(orbit_heatmap(pm["contact_min"], "Ground-station contact per day", "min", "Teal"), width="stretch")
    st.caption("Why the dawn-dusk plane (06:00 / 18:00) is not automatically best here: it removes eclipses, but the Sun "
               "then shines edge-on to the zenith and ram/wake body panels, so the array collects far less energy. "
               "Contact time depends mostly on inclination: near-polar planes see the Svalbard station on every orbit.")

    st.markdown("**Full optimisation.** Runs the complete decision engine (battery, heat, storage, real NASA scenes) "
                f"on {len(INC_OPT) * len(LTAN_OPT)} candidate orbits with your sidebar settings. About 1 minute the first time.")
    if st.button("Find the best orbit", type="primary"):
        st.session_state["opt_key"] = (BASE_PJ, V, zeta)
    if st.session_state.get("opt_key") == (BASE_PJ, V, zeta):
        opt = optimise_orbit(BASE_PJ, V, zeta, float(REAL["alt"]))
        healthy = (opt["shed_pct"] < 1.0) & (opt["soc_min_pct"] >= 50.0)
        if not healthy.any():
            healthy = opt["shed_pct"] < 1.0
        score = np.where(healthy, opt["value"], -1)
        bi, bj = np.unravel_index(np.argmax(score), score.shape)
        o1, o2 = st.columns(2)
        o1.plotly_chart(orbit_heatmap(opt["value"], "Semantic value delivered per day", "value", "Viridis",
                                      x=LTAN_OPT, y=INC_OPT), width="stretch")
        o2.plotly_chart(orbit_heatmap(opt["soc_min_pct"], "Lowest battery level over the day", "%", "RdYlGn",
                                      x=LTAN_OPT, y=INC_OPT), width="stretch")
        best_inc, best_lt = INC_OPT[bi], LTAN_OPT[bj]
        vi, vj = np.unravel_index(np.argmax(np.where(opt["shed_pct"] < 1.0, opt["value"], -1)), opt["value"].shape)
        st.success(f"Best orbit that keeps the battery above 50%: inclination {best_inc:g}°, ascending node "
                   f"{fmt_lt(best_lt)} local time. Value {opt['value'][bi, bj]:.1f}/day, lowest battery "
                   f"{opt['soc_min_pct'][bi, bj]:.0f}%, no load-shedding.")
        if (vi, vj) != (bi, bj):
            st.caption(f"Highest value regardless of battery margin: {INC_OPT[vi]:g}°, node {fmt_lt(LTAN_OPT[vj])} "
                       f"({opt['value'][vi, vj]:.1f}/day) but the battery falls to {opt['soc_min_pct'][vi, vj]:.0f}%. "
                       "Value also depends on what each orbit overflies that day: real fire alerts count triple.")

        def _apply(a=best_inc, b=best_lt):
            st.session_state["orbit_choice"] = "What-if orbit"
            st.session_state["inc"] = float(a)
            st.session_state["ltan"] = float(b)

        st.button("Fly this orbit", on_click=_apply)


# ---------------------------------------------------------------------------
# Telemetry tab
# ---------------------------------------------------------------------------
def eclipse_spans(t, sunlit):
    ecl = np.asarray(sunlit) < 0.5
    edges = np.diff(np.r_[0, ecl.astype(int), 0])
    return [(t[s0], t[min(e0, len(t) - 1)]) for s0, e0 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


with tabs[1]:
    t = tel["t_h"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.29, 0.29, 0.29, 0.13],
                        subplot_titles=("Battery state of charge [%]", "Node temperature [°C]", "Queue backlog q(t) [MB]",
                                        "E-LOTUS active action"))
    for a, b in eclipse_spans(t, tel["sunlit"]):
        for row in (1, 2, 3):
            fig.add_vrect(x0=a, x1=b, fillcolor="#1a1a19", opacity=0.07, line_width=0, row=row, col=1)
    names = ["E-LOTUS"] + compare_with
    for nm in names:
        tt = runs[nm]["telemetry"]
        col = POLICY_COLORS[nm]
        w = 2.2 if nm == "E-LOTUS" else 1.4
        fig.add_trace(go.Scatter(x=t, y=tt["soc"], name=nm, line=dict(color=col, width=w), legendgroup=nm), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=tt["temp_c"], name=nm, line=dict(color=col, width=w), legendgroup=nm, showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=tt["q_mb"], name=nm, line=dict(color=col, width=w), legendgroup=nm, showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=tel["E_min_wh"] / cfg.battery_wh * 100, name="E_min(t) eclipse-aware floor",
                             line=dict(color="#52514e", dash="dash", width=1)), 1, 1)
    fig.add_hline(y=cfg.e_crit_wh / cfg.battery_wh * 100, line=dict(color="#e34948", dash="dot", width=1), row=1, col=1,
                  annotation_text="E_crit (load-shed)", annotation_position="bottom right")
    fig.add_hline(y=cfg.t_safe_c, line=dict(color="#52514e", dash="dash", width=1), row=2, col=1,
                  annotation_text="T_safe", annotation_position="top right")
    fig.add_hline(y=cfg.t_max_c, line=dict(color="#e34948", dash="dot", width=1), row=2, col=1,
                  annotation_text="T_max (throttle)", annotation_position="top right")
    fig.add_hline(y=cfg.storage_mb, line=dict(color="#e34948", dash="dot", width=1), row=3, col=1,
                  annotation_text="storage limit", annotation_position="top right")
    act = tel["action"].astype(int)
    for code, a in enumerate(ACTIONS):
        mk = act == code
        fig.add_trace(go.Bar(x=t[mk], y=np.ones(mk.sum()), marker_color=ACTION_COLORS[a], marker_line_width=0, name=a,
                             width=cfg.dt / 3600, legendgroup="act"), 4, 1)
    fig.update_yaxes(showticklabels=False, row=4, col=1)
    fig.update_xaxes(title_text="Mission elapsed time [h] (shaded = eclipse)", row=4, col=1)
    fig.update_layout(height=860, bargap=0, hovermode="x unified", margin=dict(l=10, r=10, t=90, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Panels share the time axis; each quantity keeps its own scale (no dual y-axes).")


# ---------------------------------------------------------------------------
# Pareto tab
# ---------------------------------------------------------------------------
with tabs[2]:
    with st.spinner("Sweeping V ..."):
        sweep = {Vv: simulate(PJ, f"E-LOTUS V={Vv:g}", Vv, zeta)["metrics"] for Vv in V_GRID}
    rows = [dict(name=f"E-LOTUS V={Vv:g}", family="E-LOTUS", V=Vv, value=mm["value"], energy=mm["energy_payload_wh"],
                 latency=mm["latency_mean_min"]) for Vv, mm in sweep.items()]
    rows += [dict(name=b, family=b, V=np.nan, value=runs[b]["metrics"]["value"], energy=runs[b]["metrics"]["energy_payload_wh"],
                  latency=runs[b]["metrics"]["latency_mean_min"]) for b in BASELINES]
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        f2 = go.Figure()
        e = df[df.family == "E-LOTUS"].sort_values("energy")
        f2.add_trace(go.Scatter(x=e.energy, y=e.value, mode="lines+markers", text=[f"V={x:g}" for x in e.V],
                                line=dict(color=POLICY_COLORS["E-LOTUS"], width=2), marker=dict(size=9),
                                name="E-LOTUS (V sweep)", customdata=e.latency,
                                hovertemplate="%{text}<br>energy %{x:.0f} Wh<br>value %{y:.1f}<br>latency %{customdata:.0f} min"))
        cur = df[(df.family == "E-LOTUS") & (df.V == V)]
        f2.add_trace(go.Scatter(x=cur.energy, y=cur.value, mode="markers", name=f"current V={V:g}",
                                marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#0b0b0b", width=2))))
        symbols = {"Always-Transmit": "square", "Greedy-Process": "triangle-up", "Store-and-Forward": "diamond", "Rule-Based": "cross"}
        for b in BASELINES:
            r_ = df[df.family == b]
            f2.add_trace(go.Scatter(x=r_.energy, y=r_.value, mode="markers+text", text=[b],
                                    textposition="middle right" if b in ("Always-Transmit", "Store-and-Forward") else "bottom center",
                                    marker=dict(size=13, symbol=symbols[b], color=POLICY_COLORS[b], line=dict(color="white", width=1.5)),
                                    name=b, customdata=r_.latency,
                                    hovertemplate=b + "<br>energy %{x:.0f} Wh<br>value %{y:.1f}<br>latency %{customdata:.0f} min"))
        pad = 0.08 * (df.energy.max() - df.energy.min() + 1)
        f2.update_xaxes(range=[df.energy.min() - pad, df.energy.max() + pad])
        f2.update_yaxes(range=[min(0, df.value.min()) - 10, df.value.max() * 1.08])
        f2.update_layout(height=480, title="Pareto frontier: semantic value vs energy (hover for V)",
                         xaxis_title="Payload energy consumed [Wh]", yaxis_title="Semantic value delivered",
                         legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f2, width="stretch")
    with c2:
        f3 = go.Figure()
        f3.add_trace(go.Scatter3d(x=e.energy, y=e.latency, z=e.value, mode="lines+markers+text", text=[f"V={x:g}" for x in e.V],
                                  line=dict(color=POLICY_COLORS["E-LOTUS"], width=5), marker=dict(size=5), name="E-LOTUS"))
        for b in BASELINES:
            r_ = df[df.family == b]
            f3.add_trace(go.Scatter3d(x=r_.energy, y=r_.latency, z=r_.value, mode="markers+text", text=[b],
                                      marker=dict(size=7, color=POLICY_COLORS[b]), name=b))
        f3.update_layout(height=480, title="3D trade-off: value x energy x latency", margin=dict(l=0, r=0, t=40, b=0),
                         scene=dict(xaxis_title="Energy [Wh]", yaxis_title="Latency [min]", zaxis_title="Value"),
                         legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(f3, width="stretch")
    st.dataframe(df.drop(columns="family").round(2), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Live decision feed
# ---------------------------------------------------------------------------
def badge(a):
    return (f"<span style='background:{ACTION_COLORS[a]};color:#fff;padding:1px 7px;border-radius:4px;"
            f"font-family:monospace;font-size:12px'>{a}</span>")


def feed_html(ev):
    lines = []
    for e_ in ev:
        lines.append(f"<div style='font-family:monospace;font-size:12px;padding:2px 0;white-space:nowrap'>"
                     f"T+{e_['t_h']:06.3f} h &nbsp;{badge(e_['action'])}&nbsp; #{e_['task_id']:05d} "
                     f"{e_['kind']} H={e_['entropy']:.2f} p={e_['priority']} {e_['size_mb']:.0f} MB "
                     f"| SOC {e_['soc']:.0f}% {e_['temp_c']:.1f}°C {e_['detail']}</div>")
    return "".join(lines)


with tabs[3]:
    ev = pd.DataFrame(el["events"])
    left, right = st.columns([1, 2])
    with left:
        show = st.multiselect("Actions", list(ACTIONS), default=["PROCESS", "TRANSMIT", "OFFLOAD_ISLL"])
        win = st.slider("Time window [h]", 0.0, float(horizon), (0.0, float(horizon)), 0.25)
        counts = ev[ev.detail != "arrival"].action.value_counts() if len(ev) else pd.Series(dtype=int)
        st.markdown("**Dispatch counts**")
        for a in ACTIONS[1:]:
            st.markdown(f"{badge(a)} &nbsp; {int(counts.get(a, 0))}", unsafe_allow_html=True)
        st.markdown(f"{badge('STORE')} &nbsp; {int((ev.detail == 'arrival').sum())} arrivals buffered",
                    unsafe_allow_html=True)
        pace = st.select_slider("Replay speed", options=["Slow", "Normal", "Fast"], value="Slow")
        play = st.button("Replay live feed")
    with right:
        sel = ev[ev.action.isin(show) & ev.t_h.between(*win)] if len(ev) else ev
        box = st.empty()
        if play and len(sel):
            frames = np.linspace(sel.t_h.min(), sel.t_h.max(), 160)
            for tf in frames:
                cur = sel[sel.t_h <= tf].tail(14).iloc[::-1]
                box.markdown(feed_html(cur.to_dict("records")), unsafe_allow_html=True)
                time.sleep({"Slow": 0.35, "Normal": 0.15, "Fast": 0.05}[pace])
        else:
            box.markdown(feed_html(sel.tail(14).iloc[::-1].to_dict("records")), unsafe_allow_html=True)
    st.markdown("**Full decision log**")
    if len(sel):
        styled = sel[["t_h", "action", "task_id", "kind", "scene", "entropy", "priority", "size_mb", "w_est", "score",
                      "lat", "lon", "soc", "temp_c", "detail"]].style.map(
            lambda a: f"background-color:{ACTION_COLORS.get(a, '#fff')};color:white", subset=["action"]).format(
            {"t_h": "{:.3f}", "score": "{:.3g}", "w_est": "{:.3f}"})
        st.dataframe(styled, width="stretch", height=380, hide_index=True)
        st.caption("'scene' is ground truth derived from real MODIS cloud fraction, OSM land/water and VIIRS fire detections, shown for evaluation only; the engine sees only entropy H and the fire-trigger priority flag.")


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------
with tabs[4]:
    cols = {"value": "Value", "ceiling_pct": "% of ceiling", "completion_pct": "Completion %", "alerts_on_time_pct": "Alerts on time %",
            "energy_payload_wh": "Payload energy Wh", "value_per_wh": "Value/Wh", "latency_mean_min": "Latency min",
            "soc_min_pct": "Min SOC %", "temp_max_c": "Max T °C", "shed_pct": "Load-shed %", "throttled_pct": "Throttled %",
            "gpu_duty_pct": "GPU duty %", "dl_util_pct": "Downlink util %", "overflow": "Overflow drops", "offloaded": "ISLL offloads"}
    tab = pd.DataFrame({nm: {v: r["metrics"][k_] for k_, v in cols.items()} for nm, r in runs.items()}).T
    st.dataframe(tab.round(2), width="stretch")
    path = os.path.join(HERE, "results", "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            summ = json.load(f)
        st.markdown(f"**Published {summ['days']}-day benchmark (per-day averages, seed {summ['main_seed']})** "
                    "-- produced by `python benchmark.py`")
        t30 = pd.DataFrame({p: {v: mm[k_] for k_, v in cols.items()} for p, mm in summ["main"].items()}).T
        st.dataframe(t30.round(2), width="stretch")


# ---------------------------------------------------------------------------
# Model tab
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("**State vector**")
    st.latex(r"\mathbf{x}(t)=[E_b(t),\,g(t),\,m(t),\,T(t),\,c(t),\,q(t),\,\bar H(t),\,\Delta_{\rm AoI}(t)]^T")
    st.markdown("**Semantic entropy and value estimate** (8-bit quick-look thumbnail)")
    st.latex(r"H(S_i)=-\sum_{x=0}^{255}P(x)\log_2P(x),\qquad \hat w_i=p_i\,\big(\varepsilon+(1-\varepsilon)\,\ell_i\big)\,(H_i/8)^{\alpha\,o_i}")
    st.markdown("**Lyapunov function** (normalised deficits)")
    st.latex(r"L(\Theta)=\tfrac12\Big[\big(\tfrac{q}{Q}\big)^2+\big(\tfrac{[E_{\min}(t)-E_b]^+}{\Delta E}\big)^2+\big(\tfrac{[T-T_{\rm safe}]^+}{\Delta T}\big)^2\Big]")
    st.markdown("**Per-slot decision** (rate-normalised drift-plus-penalty, STORE is the zero reference)")
    st.latex(r"a_i^\star=\arg\min_{a\in\mathcal A}\;\frac{\widehat{\Delta L}_i(a)-V\,U_i(a)}{\tau_i(a)},\quad "
             r"U_i(a)=\hat w_i\,\eta_a\,\phi_i(\text{age}+d_a)-\zeta\,\mathcal E_i(a)")
    st.markdown("**Physics integrated with RK4 every slot**")
    st.latex(r"\dot E_b=\begin{cases}\eta_c\min(P_{\rm sol}-P_{\rm load},P_{\rm ch}(SOC))&P_{\rm sol}\ge P_{\rm load}\\(P_{\rm sol}-P_{\rm load})/\eta_d&\text{otherwise}\end{cases}"
             r"\qquad C\dot T=P_{\rm load}+Q_{\rm env}-k\,\epsilon\sigma A\,(T^4-T_{\rm space}^4)")
    st.caption("Orbit: real tracked satellite TLE (CelesTrak) propagated with SGP4; an independent RK4 two-body+J2 integrator agrees "
               "within ~2 km over 24 h. Eclipse: cylindrical shadow. Contacts: 10 deg mask at KSAT SvalSat and ASI Matera.")
    st.markdown("**Real data sources (NASA GIBS, sampled along the real ground track, 2026-08-26 to 2026-09-14)**")
    st.markdown("- Optical quick-look: `VIIRS_NOAA20_CorrectedReflectance_TrueColor`  \n"
                "- Thermal-IR quick-look (eclipse): `VIIRS_NOAA20_Brightness_Temp_BandI5_Night`  \n"
                "- Fire alerts: `VIIRS_NOAA20_Thermal_Anomalies_375m_All` (NASA FIRMS)  \n"
                "- Cloud fraction (ground truth): `MODIS_Aqua_Cloud_Fraction_Day/Night`  \n"
                "- Land fraction (ground truth): `OSM_Land_Water_Map`")
