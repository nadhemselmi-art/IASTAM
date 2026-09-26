"""
E-LOTUS simulation engine
=========================
Entropy-Weighted Lyapunov Optimization with Thermal Radiation & Swarm Offloading
IASTAM 6.0 Technical Challenge -- Track 1, Problem 1 ("Process or Transmit?")

Contents
--------
* Physical constants and ``SimConfig`` (every numeric assumption lives here).
* Real orbit           : PHISAT-2 (NORAD 60470) CelesTrak TLE propagated with SGP4
                         (``orbit_model="sgp4"``, default); ``OrbitPropagator`` gives an
                         independent RK4 two-body+J2 integration used for validation.
* Real Earth data       : ``RealAtlas`` -- per-30-s along-track features sampled from NASA GIBS
                         (VIIRS NOAA-20 true colour / I5 night BT, VIIRS thermal anomalies,
                         MODIS Aqua cloud fraction, OSM land/water) for 2026-08-26 .. 2026-09-24.
* ``Ephemeris``         : policy-independent geometry (sun, eclipse, solar incidence,
                          ground-station contacts, inter-satellite line of sight).
* ``Task``              : one unit of instrument data (dataclass).
* ``generate_workload`` : synthetic thumbnails -> Shannon entropy H(S_i) per task.
* ``SatelliteNode``     : RK4 integration of battery energy E_b(t) and radiative
                          temperature T(t); solar incidence theta(t); ISLL availability.
* ``ELotusEngine``      : drift-plus-penalty controller over the action set
                          {PROCESS, STORE, TRANSMIT, OFFLOAD_ISLL}.
* Baselines             : Always-Transmit, Greedy-Process, Store-and-Forward, Rule-Based.
* ``Simulation``        : slotted simulation loop returning full time-series logs.

Only numpy is required.
"""
from __future__ import annotations

import heapq
import math
import datetime as _dt
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418          # km^3 s^-2
R_EARTH = 6378.137              # km
J2 = 1.08262668e-3              # Earth oblateness coefficient
SIGMA_SB = 5.670374419e-8       # W m^-2 K^-4
KELVIN = 273.15

ACTIONS = ("STORE", "PROCESS", "TRANSMIT", "OFFLOAD_ISLL")
ACTION_CODE = {a: i for i, a in enumerate(ACTIONS)}
RESOURCE_ACTION = {"gpu": "PROCESS", "dl": "TRANSMIT", "isl": "OFFLOAD_ISLL"}

# Scene catalogue: occurrence probability and ground-truth scientific value.
# The engine never sees the scene label -- only the thumbnail entropy and the
# instrument priority flag. Ground truth is used exclusively for scoring.
SCENES = {
    #            prob   true value
    "ocean":    (0.40, 0.05),
    "cloud":    (0.28, 0.02),
    "land":     (0.20, 0.45),
    "urban":    (0.08, 0.65),
    "anomaly":  (0.04, 3.00),   # thermal anomaly (fire, flare, volcanic hot spot)
}

# Real ground-station sites (lat, lon in degrees).
GROUND_STATIONS = {
    "Svalbard (KSAT SvalSat)": (78.2297, 15.3975),
    "Matera (ASI CGS)": (40.6486, 16.7046),
}
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
DATA_DIR = __import__("os").path.join(HERE, "data")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    # --- time ---
    epoch: Tuple[int, int, int, int, int, int] = (2026, 8, 26, 0, 0, 0)   # start of the real-data window
    start_day: int = 0                  # offset (days) into the real 30-day window
    duration_h: float = 24.0
    dt: float = 10.0                    # slot length [s]
    seed: int = 7

    # --- orbit ---
    orbit_model: str = "sgp4"           # "sgp4" = real PHISAT-2 TLE; "rk4_j2" = idealised circular SSO
    tle_path: str = ""                  # default: data/phisat2_tle.txt
    workload: str = "real"              # "real" = NASA GIBS atlas; "synthetic" = procedural thumbnails
    atlas_path: str = ""                # default: data/elotus_gibs_atlas.bin
    fire_trigger_px: int = 128          # TIR trigger: >= 12.5 % of the quick-look flagged by VIIRS 375 m fires
    estimator: str = "entropy_land"     # "entropy_land" (default), "entropy" (raw), "land" (prior only)
    altitude_km: float = 500.0
    inclination_deg: float = 97.40
    ltan_h: float = 22.5                # local time of ascending node [h]
    elevation_mask_deg: float = 10.0
    ground_stations: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(GROUND_STATIONS))

    # --- power ---
    battery_wh: float = 100.0           # LiFePO4 pack
    soc0: float = 0.70
    e_crit_wh: float = 20.0             # hard floor -> payload load-shedding
    e_reserve_wh: float = 30.0          # policy reserve (soft floor, E-LOTUS)
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    p_charge_max_w: float = 40.0        # 0.4 C charge limit, CV taper above 90 % SOC
    p_face_w: float = 20.0              # per body-mounted face at normal incidence
    p_bus_w: float = 5.0                # OBC, ADCS, EPS, receivers
    p_gpu_idle_w: float = 1.5
    p_gpu_active_w: float = 15.0        # Jetson Orin-class module, 15 W mode
    p_tx_w: float = 12.0                # S/X-band transmitter (DC input)
    p_isl_w: float = 8.0                # optical ISL terminal (DC input)

    # --- compute ---
    gpu_gflops: float = 1000.0          # effective sustained GFLOP/s (derated)
    gflop_per_mb: float = 900.0         # onboard pipeline cost per MB of raw data
    product_ratio: float = 0.03         # product size / raw size after processing
    eta_process: float = 0.90           # fraction of information kept by a product

    # --- thermal (lumped node, radiator to deep space) ---
    emissivity: float = 0.85
    radiator_area_m2: float = 0.045
    k_thermal: float = 1.0              # dissipation coefficient multiplier (slider)
    heat_capacity_j_per_k: float = 1500.0
    q_sun_w: float = 6.0                # absorbed solar + albedo when sunlit
    q_earth_ir_w: float = 3.0           # absorbed Earth IR (always)
    t_space_k: float = 3.0
    t0_c: float = 15.0
    t_safe_c: float = 45.0              # soft limit (Lyapunov thermal deficit)
    t_max_c: float = 60.0               # hard limit -> GPU throttled

    # --- links ---
    downlink_mbps: float = 10.0
    isl_mbps: float = 100.0
    isl_enabled: bool = True
    isl_phase_deg: float = 30.0         # leading / trailing neighbour in-plane phase
    isl_graze_km: float = 100.0         # minimum ray altitude for line of sight
    nb_spare_gflops: float = 400.0      # donor spare compute while sunlit
    nb_bucket_gflop: float = 60000.0    # donor spare-compute token bucket

    # --- storage ---
    storage_mb: float = 8000.0

    # --- workload ---
    lam_sunlit: float = 1.0 / 45.0      # task arrivals per second (sunlit)
    lam_eclipse: float = 1.0 / 120.0    # task arrivals per second (eclipse, TIR only)
    size_median_mb: float = 50.0
    size_sigma: float = 0.40
    deadline_alert_s: float = 5400.0
    deadline_tasked_s: float = 6 * 3600.0
    deadline_routine_s: float = 12 * 3600.0
    tasked_fraction: float = 0.10
    load_scale: float = 1.0             # multiplies both arrival rates (stress tests)

    # --- derived helpers ---
    @property
    def n_slots(self) -> int:
        return int(round(self.duration_h * 3600.0 / self.dt))

    @property
    def epoch_jd(self) -> float:
        e = _dt.datetime(*self.epoch) + _dt.timedelta(days=self.start_day)
        j2000 = _dt.datetime(2000, 1, 1, 12, 0, 0)
        return 2451545.0 + (e - j2000).total_seconds() / 86400.0

    @property
    def rad_coeff(self) -> float:        # epsilon * sigma * A * k   [W K^-4]
        return self.emissivity * SIGMA_SB * self.radiator_area_m2 * self.k_thermal

    @property
    def dl_mb_per_s(self) -> float:
        return self.downlink_mbps / 8.0

    @property
    def isl_mb_per_s(self) -> float:
        return self.isl_mbps / 8.0

    @property
    def p_idle_w(self) -> float:
        return self.p_bus_w + self.p_gpu_idle_w


# ---------------------------------------------------------------------------
# Orbit propagation (RK4, two-body + J2)
# ---------------------------------------------------------------------------
def _accel(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    k = -MU_EARTH / (r2 * r)
    f = 1.5 * J2 * MU_EARTH * R_EARTH * R_EARTH / (r2 * r2 * r)
    zz = 5.0 * z * z / r2
    return (k * x + f * x * (zz - 1.0),
            k * y + f * y * (zz - 1.0),
            k * z + f * z * (zz - 3.0))


class OrbitPropagator:
    """Fixed-step 4th-order Runge-Kutta integration of the J2-perturbed orbit."""

    @staticmethod
    def initial_state(a_km: float, inc_deg: float, raan_deg: float, u_deg: float = 0.0):
        i, O, u = map(math.radians, (inc_deg, raan_deg, u_deg))
        v = math.sqrt(MU_EARTH / a_km)
        cO, sO, ci, si, cu, su = math.cos(O), math.sin(O), math.cos(i), math.sin(i), math.cos(u), math.sin(u)
        r = (a_km * (cO * cu - sO * su * ci), a_km * (sO * cu + cO * su * ci), a_km * su * si)
        vv = (v * (-cO * su - sO * cu * ci), v * (-sO * su + cO * cu * ci), v * cu * si)
        return r, vv

    @staticmethod
    def propagate(r0, v0, dt: float, n: int) -> np.ndarray:
        out = np.empty((n, 6))
        x, y, z = r0
        vx, vy, vz = v0
        h = dt
        for k in range(n):
            out[k, 0] = x; out[k, 1] = y; out[k, 2] = z
            out[k, 3] = vx; out[k, 4] = vy; out[k, 5] = vz
            a1 = _accel(x, y, z)
            x2, y2, z2 = x + 0.5 * h * vx, y + 0.5 * h * vy, z + 0.5 * h * vz
            vx2, vy2, vz2 = vx + 0.5 * h * a1[0], vy + 0.5 * h * a1[1], vz + 0.5 * h * a1[2]
            a2 = _accel(x2, y2, z2)
            x3, y3, z3 = x + 0.5 * h * vx2, y + 0.5 * h * vy2, z + 0.5 * h * vz2
            vx3, vy3, vz3 = vx + 0.5 * h * a2[0], vy + 0.5 * h * a2[1], vz + 0.5 * h * a2[2]
            a3 = _accel(x3, y3, z3)
            x4, y4, z4 = x + h * vx3, y + h * vy3, z + h * vz3
            vx4, vy4, vz4 = vx + h * a3[0], vy + h * a3[1], vz + h * a3[2]
            a4 = _accel(x4, y4, z4)
            x += h / 6.0 * (vx + 2 * vx2 + 2 * vx3 + vx4)
            y += h / 6.0 * (vy + 2 * vy2 + 2 * vy3 + vy4)
            z += h / 6.0 * (vz + 2 * vz2 + 2 * vz3 + vz4)
            vx += h / 6.0 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
            vy += h / 6.0 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            vz += h / 6.0 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        return out


# ---------------------------------------------------------------------------
# Ephemeris (policy-independent geometry)
# ---------------------------------------------------------------------------
def sun_unit_vector(jd: np.ndarray) -> np.ndarray:
    """Low-precision solar ephemeris (Astronomical Almanac), ECI unit vectors."""
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def gmst_rad(jd: np.ndarray) -> np.ndarray:
    return np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


def _in_shadow(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Cylindrical Earth-shadow model."""
    proj = np.einsum("ij,ij->i", r, s)
    perp = r - proj[:, None] * s
    return (proj < 0.0) & (np.linalg.norm(perp, axis=1) < R_EARTH)


def _gs_visible(r: np.ndarray, theta: np.ndarray, lat_deg: float, lon_deg: float, mask_deg: float):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    ang = lon + theta
    g = R_EARTH * np.stack([math.cos(lat) * np.cos(ang), math.cos(lat) * np.sin(ang),
                            np.full_like(ang, math.sin(lat))], axis=-1)
    rho = r - g
    up = g / R_EARTH
    sin_el = np.einsum("ij,ij->i", rho, up) / np.linalg.norm(rho, axis=1)
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    return el >= mask_deg, el


def _next_true_wait(flag: np.ndarray, dt: float) -> np.ndarray:
    """Seconds from slot k until the next slot where flag is True (0 if already True)."""
    n = len(flag)
    idx = np.flatnonzero(flag)
    ar = np.arange(n)
    if idx.size == 0:
        return np.full(n, 1e9)
    pos = np.searchsorted(idx, ar)
    wait = np.full(n, 1e9)
    ok = pos < idx.size
    wait[ok] = (idx[pos[ok]] - ar[ok]) * dt
    return wait


def _segment_clear(r1: np.ndarray, r2: np.ndarray, rmin: float) -> np.ndarray:
    d = r2 - r1
    s = np.clip(-np.einsum("ij,ij->i", r1, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    p = r1 + s[:, None] * d
    return np.linalg.norm(p, axis=1) > rmin


@dataclass
class Ephemeris:
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    sun: np.ndarray
    sunlit: np.ndarray
    theta_deg: np.ndarray           # solar incidence angle on the zenith panel
    solar_factor: np.ndarray        # sum of max(0, n_i . s) over lit faces
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    contact: np.ndarray
    contact_station: np.ndarray     # index into station list, -1 if none
    wait_contact: np.ndarray
    nb_sunlit: Dict[str, np.ndarray]
    isl_los: Dict[str, np.ndarray]
    nb_wait_contact: Dict[str, np.ndarray]
    ecl_remaining_s: np.ndarray     # if in eclipse: time to exit
    t_to_eclipse_s: np.ndarray      # if sunlit: time to next eclipse entry
    next_ecl_dur_s: np.ndarray      # if sunlit: duration of next eclipse
    station_names: List[str]
    beta_deg: np.ndarray
    dt: float


def read_tle(cfg: SimConfig) -> Tuple[str, str, str]:
    path = cfg.tle_path or __import__("os").path.join(DATA_DIR, "phisat2_tle.txt")
    lines = [ln.rstrip() for ln in open(path) if ln.strip()]
    return lines[0], lines[1], lines[2]


def propagate_tle(cfg: SimConfig, jd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SGP4 propagation of the real TLE; TEME positions are rotated to Earth-fixed with GMST."""
    from sgp4.api import Satrec
    _, l1, l2 = read_tle(cfg)
    sat = Satrec.twoline2rv(l1, l2)
    whole = np.floor(jd - 0.5) + 0.5
    e, r, v = sat.sgp4_array(whole, jd - whole)
    if np.any(e != 0):
        raise RuntimeError(f"SGP4 error codes {set(e[e != 0].tolist())}")
    return r, v


def build_ephemeris(cfg: SimConfig, extra_h: float = 14.0) -> Ephemeris:
    """extra_h: look-ahead beyond the horizon (> longest AoI deadline) for contact/eclipse prediction."""
    n = int(round((cfg.duration_h + extra_h) * 3600.0 / cfg.dt))
    t = np.arange(n) * cfg.dt
    jd = cfg.epoch_jd + t / 86400.0
    sun = sun_unit_vector(jd)
    if cfg.orbit_model == "sgp4":
        r, v = propagate_tle(cfg, jd)
    else:
        ra_sun = math.degrees(math.atan2(sun[0, 1], sun[0, 0]))
        raan = (ra_sun + (cfg.ltan_h - 12.0) * 15.0) % 360.0
        a = R_EARTH + cfg.altitude_km
        r0, v0 = OrbitPropagator.initial_state(a, cfg.inclination_deg, raan, 0.0)
        st = OrbitPropagator.propagate(r0, v0, cfg.dt, n)
        r, v = st[:, :3], st[:, 3:]
    rn = np.linalg.norm(r, axis=1)
    rhat = r / rn[:, None]
    h = np.cross(r, v)
    hhat = h / np.linalg.norm(h, axis=1)[:, None]
    that = np.cross(hhat, rhat)                    # along-track (ram) direction
    shadow = _in_shadow(r, sun)
    sunlit = ~shadow
    cz = np.einsum("ij,ij->i", rhat, sun)
    ct = np.einsum("ij,ij->i", that, sun)
    solar_factor = (np.maximum(cz, 0) + np.maximum(ct, 0) + np.maximum(-ct, 0)) * sunlit
    theta = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    beta = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", hhat, sun), -1, 1)))
    gm = gmst_rad(jd)
    lat = np.degrees(np.arcsin(r[:, 2] / rn))
    lon = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gm) + 180.0) % 360.0 - 180.0
    alt = rn - R_EARTH

    names = list(cfg.ground_stations.keys())
    contact = np.zeros(n, bool)
    station = np.full(n, -1)
    for j, nm in enumerate(names):
        vis, _ = _gs_visible(r, gm, *cfg.ground_stations[nm], cfg.elevation_mask_deg)
        station[vis & ~contact] = j
        contact |= vis
    wait_contact = _next_true_wait(contact, cfg.dt)

    phi = math.radians(cfg.isl_phase_deg)
    hxr = np.cross(hhat, r)
    nb_pos = {"lead": r * math.cos(phi) + hxr * math.sin(phi),
              "trail": r * math.cos(phi) - hxr * math.sin(phi)}
    nb_sunlit, isl_los, nb_wait = {}, {}, {}
    for side, rp in nb_pos.items():
        nb_sunlit[side] = ~_in_shadow(rp, sun)
        isl_los[side] = _segment_clear(r, rp, R_EARTH + cfg.isl_graze_km)
        c = np.zeros(n, bool)
        for nm in names:
            vis, _ = _gs_visible(rp, gm, *cfg.ground_stations[nm], cfg.elevation_mask_deg)
            c |= vis
        nb_wait[side] = _next_true_wait(c, cfg.dt)

    # eclipse timing
    ecl_remaining = _next_true_wait(sunlit, cfg.dt) * shadow
    t_to_ecl = _next_true_wait(shadow, cfg.dt) * sunlit
    # beyond the end of the look-ahead window the wait is unknown: clip to one orbital period
    ecl_remaining = np.where(ecl_remaining > 1e8, 0.0, ecl_remaining)
    t_to_ecl = np.where(t_to_ecl > 1e8, 0.0, t_to_ecl)
    next_dur = np.zeros(n)
    # run-length of each eclipse
    edges = np.diff(np.concatenate([[0], shadow.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    dur_at_start = np.zeros(n)
    for s_, e_ in zip(starts, ends):
        dur_at_start[s_] = (e_ - s_) * cfg.dt
    nxt = np.minimum(np.arange(n) + (t_to_ecl / cfg.dt).astype(int), n - 1)
    next_dur = np.where(sunlit, dur_at_start[nxt], 0.0)

    return Ephemeris(t=t, r=r, v=v, sun=sun, sunlit=sunlit, theta_deg=theta,
                     solar_factor=solar_factor, lat=lat, lon=lon, alt=alt,
                     contact=contact, contact_station=station, wait_contact=wait_contact,
                     nb_sunlit=nb_sunlit, isl_los=isl_los, nb_wait_contact=nb_wait,
                     ecl_remaining_s=ecl_remaining, t_to_eclipse_s=t_to_ecl,
                     next_ecl_dur_s=next_dur, station_names=names, beta_deg=beta, dt=cfg.dt)


def orbit_power_map(cfg: SimConfig, incs, ltans, hours: float = 24.0, dt: float = 30.0) -> Dict[str, np.ndarray]:
    """Geometry-only sweep for the orbit designer (circular orbit, J2 secular node drift).

    For every (inclination, LTAN) pair: eclipse share, solar-array energy available per day for the
    body-mounted zenith + ram/wake faces, mean beta angle and ground-contact minutes per day.
    Fast (vectorised); the full controller simulation is run separately for the chosen orbit.
    """
    n = int(round(hours * 3600.0 / dt))
    t = np.arange(n) * dt
    jd = cfg.epoch_jd + t / 86400.0
    sun = sun_unit_vector(jd)
    gm = gmst_rad(jd)
    a = R_EARTH + cfg.altitude_km
    nmot = math.sqrt(MU_EARTH / a ** 3)
    ra_sun = math.atan2(sun[0, 1], sun[0, 0])
    u = nmot * t
    cu, su = np.cos(u), np.sin(u)
    shape = (len(incs), len(ltans))
    out = {k: np.zeros(shape) for k in ("eclipse_pct", "solar_wh", "beta_deg", "contact_min")}
    for i, inc in enumerate(incs):
        ir = math.radians(inc)
        ci, si = math.cos(ir), math.sin(ir)
        draan = -1.5 * nmot * J2 * (R_EARTH / a) ** 2 * ci
        for j, lt in enumerate(ltans):
            O = ra_sun + math.radians((lt - 12.0) * 15.0) + draan * t
            cO, sO = np.cos(O), np.sin(O)
            rhat = np.stack([cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si], axis=1)
            that = np.stack([-cO * su - sO * cu * ci, -sO * su + cO * cu * ci, cu * si], axis=1)
            hhat = np.cross(rhat, that)
            r = a * rhat
            lit = ~_in_shadow(r, sun)
            cz = np.einsum("ij,ij->i", rhat, sun)
            ct = np.einsum("ij,ij->i", that, sun)
            sf = (np.maximum(cz, 0) + np.abs(ct)) * lit
            contact = np.zeros(n, bool)
            for nm in cfg.ground_stations:
                vis, _ = _gs_visible(r, gm, *cfg.ground_stations[nm], cfg.elevation_mask_deg)
                contact |= vis
            out["eclipse_pct"][i, j] = 100.0 * (1.0 - lit.mean())
            out["solar_wh"][i, j] = cfg.p_face_w * sf.sum() * dt / 3600.0 * 24.0 / hours
            out["beta_deg"][i, j] = float(np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", hhat, sun), -1, 1))).mean())
            out["contact_min"][i, j] = contact.sum() * dt / 60.0 * 24.0 / hours
    return out


# ---------------------------------------------------------------------------
# Tasks and workload
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: int
    t_gen: float                 # generation time [s]
    size_mb: float
    gflops: float                # required compute [GFLOP]
    entropy: float               # Shannon entropy H_i of the quick-look thumbnail [bit/px]
    priority: int                # 1 routine, 2 tasked, 4 alert (instrument trigger)
    aoi_deadline_s: float        # Age-of-Information deadline
    scene: str                   # hidden ground truth (never read by the engine)
    true_value: float            # hidden ground-truth scientific value
    est_value: float = 0.0       # engine-side semantic value estimate w_i
    kind: str = "raw"            # "raw" | "product"
    parent_id: int = -1
    remaining: float = 0.0       # remaining GFLOP (processing) or MB (transfer)
    lat: float = float("nan")    # scene centre (real ground track)
    lon: float = float("nan")
    land: float = float("nan")   # real land fraction (OSM land/water)
    cloud: float = float("nan")  # real cloud fraction (MODIS Aqua)
    fire_px: int = 0             # real VIIRS thermal-anomaly pixels in the scene
    optical: bool = True         # optical (sunlit) or thermal-IR (eclipse) quick-look

    def age(self, t: float) -> float:
        return t - self.t_gen

    def freshness(self, t: float) -> float:
        a = t - self.t_gen
        return math.exp(-a / self.aoi_deadline_s) if a <= self.aoi_deadline_s else 0.0


def shannon_entropy(img: np.ndarray) -> float:
    """H(S) = -sum p(x) log2 p(x) over the 8-bit grey-level histogram."""
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def synth_tir(scene: str, rng: np.random.Generator, n: int = 16) -> np.ndarray:
    """Thermal-infrared quick-look used by the instrument hot-pixel trigger."""
    img = rng.normal(120.0, 6.0, (n, n))
    if scene == "anomaly":
        i0, j0 = rng.integers(0, n - 4, 2)
        img[i0:i0 + 4, j0:j0 + 4] = rng.uniform(240, 255, (4, 4))
    return np.clip(img, 0, 255).astype(np.uint8)


def synth_thumbnail(scene: str, rng: np.random.Generator, n: int = 32) -> np.ndarray:
    if scene == "ocean":
        img = rng.normal(38.0, 2.0, (n, n))
    elif scene == "cloud":
        img = rng.normal(226.0, 5.0, (n, n))
    elif scene == "land":
        centers = rng.choice([70.0, 105.0, 140.0], size=(n // 4, n // 4), p=[0.4, 0.35, 0.25])
        img = np.kron(centers, np.ones((4, 4))) + rng.normal(0, 8.0, (n, n))
    elif scene == "urban":
        blocks = rng.uniform(20, 230, (n // 4, n // 4))
        img = np.kron(blocks, np.ones((4, 4))) + rng.normal(0, 18.0, (n, n))
    else:  # anomaly = land background + saturated hot spot
        centers = rng.choice([70.0, 105.0, 140.0], size=(n // 4, n // 4), p=[0.4, 0.35, 0.25])
        img = np.kron(centers, np.ones((4, 4))) + rng.normal(0, 8.0, (n, n))
        i0, j0 = rng.integers(0, n - 8, 2)
        img[i0:i0 + 7, j0:j0 + 7] = rng.uniform(235, 255, (7, 7))
    return np.clip(img, 0, 255).astype(np.uint8)


def semantic_value(entropy: float, priority: int, alpha: float = 3.0, h_max: float = 8.0) -> float:
    """w_i = p_i * (H_i / H_max)^alpha  (entropy-sharpened semantic weight)."""
    return priority * (entropy / h_max) ** alpha


def semantic_estimate(entropy: float, priority: int, land_prior: float, optical: bool,
                      estimator: str = "entropy_land", eps: float = 0.1, alpha: float = 3.0) -> float:
    """Onboard value estimate from quick-look entropy and the static land-mask prior of the footprint.

    entropy_land : p * (eps + (1-eps) * land) * (H/8)^alpha for optical quick-looks,
                   p * (eps + (1-eps) * land) for thermal-IR quick-looks (night-time TIR entropy is
                   dominated by cloud-top texture and anti-correlated with value on real data)
    entropy      : p * (H/8)^alpha            (raw semantic entropy)
    land         : p * (eps + (1-eps) * land)
    """
    g = eps + (1.0 - eps) * land_prior
    if estimator == "entropy":
        return priority * (entropy / 8.0) ** alpha
    if estimator == "land":
        return priority * g
    return priority * g * ((entropy / 8.0) ** alpha if optical else 1.0)


class RealAtlas:
    """Along-track Earth features sampled from NASA GIBS every 30 s over the real 30-day window.

    Produced in-browser by ``data/gibs_sampler.js`` (see paper Sec. IV-A): for each PHISAT-2
    sub-satellite point a 32x32-pixel (~313 km) quick-look block is cut from the global 4096x2048
    daily mosaics and summarised.
    """
    _cache: Dict[str, "RealAtlas"] = {}

    def __init__(self, path: str):
        import json as _json
        raw = open(path, "rb").read()
        self.header = _json.loads(raw[:4096].decode().strip())
        n = self.header["n"]
        o = 4096
        self.lat = np.frombuffer(raw, np.float32, n, o); o += 4 * n
        self.lon = np.frombuffer(raw, np.float32, n, o); o += 4 * n
        fields = ["H_tc", "H_bt", "land", "fire", "cloud_day", "cloud_night", "mean_tc"]
        for f in fields:
            setattr(self, f, np.frombuffer(raw, np.uint8, n, o).copy()); o += n
        self.n = n
        self.step = float(self.header["step_s"])
        self.days_done = int(self.header.get("days_done", 30))

    @classmethod
    def load(cls, cfg: SimConfig) -> "RealAtlas":
        path = cfg.atlas_path or __import__("os").path.join(DATA_DIR, "elotus_gibs_atlas.bin")
        if path not in cls._cache:
            cls._cache[path] = cls(path)
        return cls._cache[path]

    def nearest(self, lat: np.ndarray, lon: np.ndarray, day: np.ndarray, k: int = 24,
                radius_deg: float = 1.5) -> np.ndarray:
        """Index of the real NASA sample that best matches (lat, lon) on a what-if ground track.

        Among the k nearest real footprints within ``radius_deg``, the one closest in date wins;
        if none is that close, the geographically nearest sample is used.
        """
        from scipy.spatial import cKDTree
        limit = min(self.n, int(self.days_done * 86400 / self.step))
        if getattr(self, "_tree", None) is None:
            la, lo = np.radians(self.lat[:limit]), np.radians(self.lon[:limit])
            xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
            self._tree = cKDTree(xyz)
            self._day = (np.arange(limit) * self.step // 86400).astype(int)
        la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
        q = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
        dist, idx = self._tree.query(q, k=k)
        chord = 2 * math.sin(math.radians(radius_deg) / 2)
        dday = np.abs(self._day[idx] - np.asarray(day)[:, None]).astype(float)
        dday[dist > chord] = 1e6
        best = np.argmin(dday + dist, axis=1)          # same-day wins; distance breaks ties
        return idx[np.arange(len(idx)), best]


def real_scene_label(land: float, cloud: float, alert: bool, dark: bool = False) -> str:
    if alert:
        return "fire"
    if dark:
        return "dark"
    if cloud >= 0.6:
        return "cloud"
    if land < 0.1:
        return "ocean"
    if land < 0.6:
        return "coast"
    return "land"


def generate_real_workload(cfg: SimConfig, eph: Ephemeris, n_slots: Optional[int] = None) -> List[List[Task]]:
    """Arrivals along the real PHISAT-2 ground track with features from real NASA imagery.

    * sunlit  -> optical quick-look entropy from VIIRS NOAA-20 corrected-reflectance true colour
    * eclipse -> thermal-IR quick-look entropy from VIIRS NOAA-20 I5 night brightness temperature
    * alert (priority 4) when the scene holds >= fire_trigger_px VIIRS 375 m thermal-anomaly pixels
    * hidden ground truth: v = 0.02 + 0.63 * land * (1 - cloud) (MODIS Aqua cloud fraction), alerts 3.0,
      0.02 for optical scenes in polar night (mean VIIRS luminance < 8)
    """
    A = RealAtlas.load(cfg)
    rng = np.random.default_rng(cfg.seed)
    n_slots = n_slots or cfg.n_slots
    off = int(round(cfg.start_day * 86400 / A.step))
    limit = min(A.n, int(A.days_done * 86400 / A.step))
    whatif = cfg.orbit_model != "sgp4"
    if whatif:   # what-if orbit: borrow the real NASA scene under each modelled footprint
        nn = A.nearest(eph.lat[:n_slots], eph.lon[:n_slots],
                       cfg.start_day + (np.arange(n_slots) * cfg.dt // 86400).astype(int))
    arrivals: List[List[Task]] = []
    tid = 0
    for k in range(n_slots):
        lam = cfg.load_scale * (cfg.lam_sunlit if eph.sunlit[k] else cfg.lam_eclipse)
        m = rng.poisson(lam * cfg.dt)
        slot: List[Task] = []
        for _ in range(m):
            t_gen = k * cfg.dt + rng.uniform(0, cfg.dt)
            i = int(nn[k]) if whatif else min(off + int(round(t_gen / A.step)), limit - 1)
            sun = bool(eph.sunlit[k])
            H = (A.H_tc[i] if sun else A.H_bt[i]) / 30.0
            cl = A.cloud_day[i] if sun else A.cloud_night[i]
            if cl == 255:
                cl = A.cloud_night[i] if sun else A.cloud_day[i]
            cloud = 0.5 if cl == 255 else cl / 100.0
            land = A.land[i] / 100.0
            alert = int(A.fire[i]) >= cfg.fire_trigger_px
            dark = sun and A.mean_tc[i] < 8          # optical scene in polar night: no usable content
            prio = 4 if alert else 1
            value = 3.0 if alert else (0.02 if dark else 0.02 + 0.63 * land * (1.0 - cloud))
            size = float(cfg.size_median_mb * math.exp(cfg.size_sigma * rng.standard_normal()))
            task = Task(task_id=tid, t_gen=t_gen, size_mb=size, gflops=size * cfg.gflop_per_mb, entropy=float(H),
                        priority=prio, aoi_deadline_s=cfg.deadline_alert_s if alert else cfg.deadline_routine_s,
                        scene=real_scene_label(land, cloud, alert, dark), true_value=value,
                        est_value=semantic_estimate(float(H), prio, land, sun, cfg.estimator))
            task.optical = sun
            task.lat, task.lon = ((float(eph.lat[k]), float(eph.lon[k])) if whatif
                                  else (float(A.lat[i]), float(A.lon[i])))
            task.land, task.cloud, task.fire_px = land, cloud, int(A.fire[i])
            slot.append(task)
            tid += 1
        arrivals.append(slot)
    return arrivals


def generate_workload(cfg: SimConfig, eph: Ephemeris, n_slots: Optional[int] = None) -> List[List[Task]]:
    """Policy-independent arrival stream: identical for every controller (same seed)."""
    if cfg.workload == "real":
        return generate_real_workload(cfg, eph, n_slots)
    return generate_synthetic_workload(cfg, eph, n_slots)


def generate_synthetic_workload(cfg: SimConfig, eph: Ephemeris, n_slots: Optional[int] = None) -> List[List[Task]]:
    """Procedural thumbnails (kept for comparison with the real-data workload)."""
    rng = np.random.default_rng(cfg.seed)
    n_slots = n_slots or cfg.n_slots
    names = list(SCENES.keys())
    probs = np.array([SCENES[s][0] for s in names])
    arrivals: List[List[Task]] = []
    tid = 0
    for k in range(n_slots):
        lam = cfg.load_scale * (cfg.lam_sunlit if eph.sunlit[k] else cfg.lam_eclipse)
        m = rng.poisson(lam * cfg.dt)
        slot: List[Task] = []
        for _ in range(m):
            scene = names[rng.choice(len(names), p=probs)]
            img = synth_thumbnail(scene, rng)
            H = shannon_entropy(img)
            hot = float((synth_tir(scene, rng) >= 235).mean())   # TIR hot-pixel trigger
            prio, value = 1, SCENES[scene][1]
            if hot > 0.02:
                prio = 4
            elif scene in ("land", "urban") and rng.random() < cfg.tasked_fraction:
                prio, value = 2, 2.0 * value            # customer-tasked acquisition
            deadline = {4: cfg.deadline_alert_s, 2: cfg.deadline_tasked_s}.get(prio, cfg.deadline_routine_s)
            size = float(cfg.size_median_mb * math.exp(cfg.size_sigma * rng.standard_normal()))
            t_gen = k * cfg.dt + rng.uniform(0, cfg.dt)
            slot.append(Task(task_id=tid, t_gen=t_gen, size_mb=size, gflops=size * cfg.gflop_per_mb,
                             entropy=H, priority=prio, aoi_deadline_s=deadline, scene=scene,
                             true_value=value, est_value=semantic_value(H, prio)))
            tid += 1
        arrivals.append(slot)
    return arrivals


# ---------------------------------------------------------------------------
# Satellite node physics
# ---------------------------------------------------------------------------
class SatelliteNode:
    """Battery + lumped radiative thermal node, integrated with RK4.

    dE_b/dt = eta_c * min(P_sol - P_load, P_ch,max(SOC))      if P_sol >= P_load
            = (P_sol - P_load) / eta_d                         otherwise
    C dT/dt = P_load + Q_env(t) - eps*sigma*A*k*(T^4 - T_space^4)
    """

    def __init__(self, cfg: SimConfig, eph: Ephemeris):
        self.cfg = cfg
        self.eph = eph
        self.E = cfg.soc0 * cfg.battery_wh          # Wh
        self.T = cfg.t0_c + KELVIN                  # K
        self.p_solar_series = cfg.p_face_w * eph.solar_factor
        self.q_env_series = cfg.q_sun_w * eph.sunlit + cfg.q_earth_ir_w

    # --- geometry accessors -------------------------------------------------
    def solar_incidence_deg(self, k: int) -> float:
        return float(self.eph.theta_deg[k])

    def in_sunlight(self, k: int) -> bool:
        return bool(self.eph.sunlit[k])

    def isl_available(self, k: int, side: str) -> bool:
        """Line of sight to the neighbour AND neighbour in full sunlight."""
        return bool(self.eph.isl_los[side][k] and self.eph.nb_sunlit[side][k])

    # --- dynamics -----------------------------------------------------------
    def _p_solar(self, k: int, frac: float) -> float:
        s = self.p_solar_series
        k1 = min(k + 1, len(s) - 1)
        return (1 - frac) * s[k] + frac * s[k1]

    def _q_env(self, k: int, frac: float) -> float:
        q = self.q_env_series
        k1 = min(k + 1, len(q) - 1)
        return (1 - frac) * q[k] + frac * q[k1]

    def _dE(self, E: float, p_sol: float, p_load: float) -> float:
        c = self.cfg
        net = p_sol - p_load
        if net >= 0.0:
            soc = E / c.battery_wh
            taper = min(1.0, max(0.0, (1.0 - soc) / 0.10))
            return c.eta_charge * min(net, c.p_charge_max_w * taper) / 3600.0
        return net / c.eta_discharge / 3600.0

    def _dT(self, T: float, q_in: float) -> float:
        c = self.cfg
        return (q_in - c.rad_coeff * (T ** 4 - c.t_space_k ** 4)) / c.heat_capacity_j_per_k

    def step(self, k: int, p_load: float, dt: float) -> None:
        """Advance (E_b, T) over one slot with classical RK4; inputs vary inside the slot."""
        E, T = self.E, self.T
        ps0, ps1, ps2 = self._p_solar(k, 0.0), self._p_solar(k, 0.5), self._p_solar(k, 1.0)
        q0, q1, q2 = self._q_env(k, 0.0), self._q_env(k, 0.5), self._q_env(k, 1.0)
        k1E, k1T = self._dE(E, ps0, p_load), self._dT(T, p_load + q0)
        k2E, k2T = self._dE(E + 0.5 * dt * k1E, ps1, p_load), self._dT(T + 0.5 * dt * k1T, p_load + q1)
        k3E, k3T = self._dE(E + 0.5 * dt * k2E, ps1, p_load), self._dT(T + 0.5 * dt * k2T, p_load + q1)
        k4E, k4T = self._dE(E + dt * k3E, ps2, p_load), self._dT(T + dt * k3T, p_load + q2)
        self.E = min(self.cfg.battery_wh, max(0.0, E + dt / 6.0 * (k1E + 2 * k2E + 2 * k3E + k4E)))
        self.T = T + dt / 6.0 * (k1T + 2 * k2T + 2 * k3T + k4T)

    # --- one-step predictors used by the controller -------------------------
    def thermal_gain(self) -> float:
        """Linearised radiative conductance G = 4 eps sigma A k T^3 [W/K]."""
        return 4.0 * self.cfg.rad_coeff * self.T ** 3

    def q_rad(self) -> float:
        c = self.cfg
        return c.rad_coeff * (self.T ** 4 - c.t_space_k ** 4)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class Policy:
    name = "policy"
    color = "#888888"

    def decide(self, sim: "Simulation", free: List[str]) -> List[Tuple[str, Task, dict]]:
        raise NotImplementedError


class ELotusEngine(Policy):
    """Entropy-weighted Lyapunov drift-plus-penalty controller.

    Lyapunov function (normalised):
        L(Theta) = 1/2 [ w_q (q/Q)^2 + w_E (D_E/dE_ref)^2 + w_T (D_T/dT_ref)^2 ]
        D_E = max(0, E_min(t) - E_b),  D_T = max(0, T - T_safe)
    For each candidate (task i, action a) with service time tau:
        Phi_i(a) = [ Delta L_i(a) - V * U_i(a) ] / tau_i(a)
    where Delta L is the *predicted exact* drift relative to STORE (Phi(STORE) = 0)
    and U = w_i * eta_a * phi(age + delay) - zeta * energy_Wh.
    Hard constraints (E > E_crit, T < T_max) define the feasible set.
    Free resources are filled greedily by the most negative Phi.
    """
    name = "E-LOTUS"
    color = "#0F766E"

    def __init__(self, V: float = 0.3, zeta: float = 0.5, w_q: float = 1.0, w_E: float = 1.0,
                 w_T: float = 1.0, dE_ref_wh: float = 5.0, dT_ref_k: float = 5.0,
                 use_entropy: bool = True, use_isl: bool = True, eclipse_aware: bool = True,
                 estimator: Optional[str] = None, label: Optional[str] = None):
        self.estimator = estimator
        self.V, self.zeta = V, zeta
        self.w_q, self.w_E, self.w_T = w_q, w_E, w_T
        self.dE_ref, self.dT_ref = dE_ref_wh, dT_ref_k
        self.use_entropy, self.use_isl, self.eclipse_aware = use_entropy, use_isl, eclipse_aware
        if label:
            self.name = label

    def _w(self, tasks: List[Task]) -> np.ndarray:
        if self.use_entropy and self.estimator is not None:
            f = 1.0 if tasks and tasks[0].kind == "raw" else 0.9
            return np.fromiter((f * semantic_estimate(t.entropy, t.priority, 0.0 if t.land != t.land else t.land,
                                                      t.optical, self.estimator) for t in tasks), float, len(tasks))
        if self.use_entropy:
            return np.fromiter((t.est_value for t in tasks), float, len(tasks))
        return np.fromiter((0.5 * t.priority for t in tasks), float, len(tasks))

    def decide(self, sim: "Simulation", free: List[str]):
        cfg, node, k = sim.cfg, sim.node, sim.k
        t_now = sim.t
        raw, prods = sim.raw_q, sim.prod_q
        if not raw and not prods:
            return []
        n_eph = len(sim.eph.t)
        Q = cfg.storage_mb
        q = sim.backlog_mb
        E, T = node.E, node.T
        Emin_arr = sim.e_min if self.eclipse_aware else sim.e_min_static
        Esurv_arr = sim.e_survive if self.eclipse_aware else sim.e_survive_static
        Tsafe = cfg.t_safe_c + KELVIN
        Tmax = cfg.t_max_c + KELVIN
        G = node.thermal_gain()
        C = cfg.heat_capacity_j_per_k
        p_sol = node.p_solar_series[k]
        p_net0 = p_sol - cfg.p_idle_w
        q_in0 = cfg.p_idle_w + node.q_env_series[k]
        dT_free = (q_in0 - node.q_rad()) / G       # asymptotic idle drift [K]
        # shunt-aware energy price: marginal energy is free while the array is being shunted
        shunting = sim.eph.sunlit[k] and p_net0 > 0 and E >= 0.95 * cfg.battery_wh
        zeta = 0.0 if shunting else self.zeta

        def drift(mu, tau, dP):
            """Predicted exact drift L(a) - L(STORE) over horizon tau (vectorised)."""
            idx = np.minimum(k + np.ceil(tau / cfg.dt).astype(int), n_eph - 1)
            Emin = Emin_arr[idx]
            eff = np.where(p_net0 > 0, cfg.eta_charge, 1.0 / cfg.eta_discharge)
            E_s = np.clip(E + eff * p_net0 * tau / 3600.0, 0.0, cfg.battery_wh)
            E_a = E_s - dP * tau / 3600.0 / cfg.eta_discharge
            decay = 1.0 - np.exp(-G * tau / C)
            T_s = T + dT_free * decay
            T_a = T_s + dP / G * decay
            dq = 0.5 * self.w_q * (((q - mu) / Q) ** 2 - (q / Q) ** 2)
            dE = 0.5 * self.w_E * ((np.maximum(0, Emin - E_a) / self.dE_ref) ** 2
                                   - (np.maximum(0, Emin - E_s) / self.dE_ref) ** 2)
            dT = 0.5 * self.w_T * ((np.maximum(0, T_a - Tsafe) / self.dT_ref) ** 2
                                   - (np.maximum(0, T_s - Tsafe) / self.dT_ref) ** 2)
            feasible = (E_a > Esurv_arr[idx] + 0.5) & (T_a < Tmax - 1.0)
            return dq + dE + dT, feasible

        def phi(age, D):
            return np.where(age <= D, np.exp(-age / D), 0.0)

        cand = []   # (score_array, resource, pool_name, extra)
        if raw:
            S = np.fromiter((t.size_mb for t in raw), float, len(raw))
            Gf = np.fromiter((t.gflops for t in raw), float, len(raw))
            age = t_now - np.fromiter((t.t_gen for t in raw), float, len(raw))
            D = np.fromiter((t.aoi_deadline_s for t in raw), float, len(raw))
            W = self._w(raw)
            if "gpu" in free:
                tau = Gf / cfg.gpu_gflops
                dP = cfg.p_gpu_active_w - cfg.p_gpu_idle_w
                idx = np.minimum(k + np.ceil(tau / cfg.dt).astype(int), n_eph - 1)
                delay = tau + sim.eph.wait_contact[idx]
                eWh = dP * tau / 3600.0
                U = W * cfg.eta_process * phi(age + delay, D) - zeta * eWh
                dL, ok = drift(S * (1 - cfg.product_ratio), tau, dP)
                score = np.where(ok, (dL - self.V * U) / tau, np.inf)
                cand.append((score, "gpu", "raw", None))
            if "dl" in free:
                tau = S / cfg.dl_mb_per_s
                eWh = cfg.p_tx_w * tau / 3600.0
                U = W * phi(age + tau, D) - zeta * eWh
                dL, ok = drift(S, tau, cfg.p_tx_w)
                score = np.where(ok, (dL - self.V * U) / tau, np.inf)
                cand.append((score, "dl", "raw", None))
            if "isl" in free and self.use_isl:
                side = sim.best_isl_side()
                if side is not None:
                    bucket = sim.nb_bucket[side]
                    tau = S / cfg.isl_mb_per_s
                    t_proc = Gf / cfg.nb_spare_gflops
                    idx = np.minimum(k + np.ceil((tau + t_proc) / cfg.dt).astype(int), n_eph - 1)
                    delay = tau + t_proc + sim.eph.nb_wait_contact[side][idx]
                    eWh = cfg.p_isl_w * tau / 3600.0
                    U = W * cfg.eta_process * phi(age + delay, D) - zeta * eWh
                    dL, ok = drift(S, tau, cfg.p_isl_w)
                    ok = ok & (Gf <= bucket)
                    score = np.where(ok, (dL - self.V * U) / tau, np.inf)
                    cand.append((score, "isl", "raw", side))
        if prods and "dl" in free:
            S = np.fromiter((t.size_mb for t in prods), float, len(prods))
            age = t_now - np.fromiter((t.t_gen for t in prods), float, len(prods))
            D = np.fromiter((t.aoi_deadline_s for t in prods), float, len(prods))
            W = self._w(prods)
            tau = S / cfg.dl_mb_per_s
            eWh = cfg.p_tx_w * tau / 3600.0
            U = W * phi(age + tau, D) - zeta * eWh
            dL, ok = drift(S, tau, cfg.p_tx_w)
            score = np.where(ok, (dL - self.V * U) / tau, np.inf)
            cand.append((score, "dl", "prod", None))

        out = []
        used_raw, used_res = set(), set()
        for _ in range(len(free)):
            best = (0.0, None)
            for ci, (score, res, pool, extra) in enumerate(cand):
                if res in used_res:
                    continue
                s = score
                if pool == "raw" and used_raw:
                    s = score.copy()
                    s[list(used_raw)] = np.inf
                j = int(np.argmin(s))
                if s[j] < best[0]:
                    best = (float(s[j]), (ci, j))
            if best[1] is None:
                break
            ci, j = best[1]
            score, res, pool, extra = cand[ci]
            task = raw[j] if pool == "raw" else prods[j]
            if pool == "raw":
                used_raw.add(j)
            used_res.add(res)
            out.append((res, task, {"side": extra, "phi": best[0]}))
        return out


class AlwaysTransmit(Policy):
    """Downlink every raw item FIFO whenever a ground station is in view."""
    name = "Always-Transmit"
    color = "#B45309"

    def decide(self, sim, free):
        if "dl" in free and sim.raw_q:
            return [("dl", sim.raw_q[0], {})]
        return []


class GreedyProcess(Policy):
    """Process every raw item FIFO as soon as the GPU is free; downlink products FIFO."""
    name = "Greedy-Process"
    color = "#7C3AED"

    def decide(self, sim, free):
        out = []
        taken = None
        if "gpu" in free and sim.raw_q:
            taken = sim.raw_q[0]
            out.append(("gpu", taken, {}))
        if "dl" in free:
            if sim.prod_q:
                out.append(("dl", sim.prod_q[0], {}))
            else:
                for t in sim.raw_q:
                    if t is not taken:
                        out.append(("dl", t, {}))
                        break
        return out


class StoreAndForward(Policy):
    """Store everything; during contacts forward by (priority, entropy) order. No onboard compute."""
    name = "Store-and-Forward"
    color = "#2563EB"

    def decide(self, sim, free):
        if "dl" in free and sim.raw_q:
            best = max(sim.raw_q, key=lambda t: (t.priority, t.entropy))
            return [("dl", best, {})]
        return []


class RuleBased(Policy):
    """Entropy-aware threshold rules (typical hand-tuned onboard rule engine)."""
    name = "Rule-Based"
    color = "#DB2777"

    def __init__(self, soc_min: float = 0.5, w_min: float = 0.10):
        self.soc_min, self.w_min = soc_min, w_min

    def decide(self, sim, free):
        cfg, node = sim.cfg, sim.node
        out, taken = [], set()
        ranked = sorted(sim.raw_q, key=lambda t: -t.est_value)
        if "dl" in free:
            if sim.prod_q:
                out.append(("dl", max(sim.prod_q, key=lambda t: t.est_value), {}))
            elif ranked:
                out.append(("dl", ranked[0], {}))
                taken.add(ranked[0].task_id)
        if "isl" in free and not sim.eph.sunlit[sim.k]:
            side = sim.best_isl_side()
            if side is not None:
                for t in ranked:
                    if t.task_id not in taken and t.est_value >= self.w_min and t.gflops <= sim.nb_bucket[side]:
                        out.append(("isl", t, {"side": side}))
                        taken.add(t.task_id)
                        break
        if ("gpu" in free and node.E / cfg.battery_wh >= self.soc_min
                and node.T < cfg.t_safe_c + KELVIN):
            for t in ranked:
                if t.task_id not in taken and t.est_value >= self.w_min:
                    out.append(("gpu", t, {}))
                    break
        return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    policy: str
    telemetry: Dict[str, np.ndarray]
    events: List[dict]
    metrics: Dict[str, float]
    deliveries: List[dict]


class Simulation:
    """Slotted simulation: arrivals -> interlocks -> service/decisions -> RK4 physics -> logs."""

    def __init__(self, cfg: SimConfig, policy: Policy, eph: Optional[Ephemeris] = None,
                 arrivals: Optional[List[List[Task]]] = None, max_events: int = 60000):
        self.cfg = cfg
        self.policy = policy
        self.eph = eph if eph is not None else build_ephemeris(cfg)
        base = arrivals if arrivals is not None else generate_workload(cfg, self.eph)
        # fresh copies so several controllers can share one arrival stream
        self.arrivals = [[replace(t) for t in slot] for slot in base]
        self.node = SatelliteNode(cfg, self.eph)
        self.raw_q: List[Task] = []
        self.prod_q: List[Task] = []
        self.jobs: Dict[str, Optional[Task]] = {"gpu": None, "dl": None, "isl": None}
        self.job_meta: Dict[str, dict] = {"gpu": {}, "dl": {}, "isl": {}}
        self.nb_bucket = {"lead": cfg.nb_bucket_gflop, "trail": cfg.nb_bucket_gflop}
        self.pending_nb: List[Tuple[float, int, Task]] = []
        self.k, self.t = 0, 0.0
        self.shed, self.throttled = False, False
        self.max_events = max_events
        self.events: List[dict] = []
        self.deliveries: List[dict] = []
        self.backlog_mb = 0.0
        self.aoi_u = 0.0         # generation time of freshest delivered update
        self.cnt = dict(generated=0, overflow=0, expired=0, processed=0, offloaded=0,
                        tx_raw=0, tx_prod=0, delivered=0, alerts=0, alerts_ok=0)
        self.value_lost_overflow = 0.0
        self._build_energy_floor()

    # --- eclipse-aware soft energy floor E_min(t) ----------------------------
    def _build_energy_floor(self):
        c, e = self.cfg, self.eph
        p_svc = c.p_idle_w + 0.5 * (c.p_gpu_active_w - c.p_gpu_idle_w)   # eclipse service load
        p_sun = float(np.mean(c.p_face_w * e.solar_factor[e.sunlit])) if e.sunlit.any() else 0.0
        need_ecl = p_svc * e.ecl_remaining_s / c.eta_discharge
        need_sun = np.maximum(0.0, p_svc * e.next_ecl_dur_s / c.eta_discharge
                              - max(0.0, p_sun - c.p_idle_w) * c.eta_charge * e.t_to_eclipse_s)
        need = np.where(e.sunlit, need_sun, need_ecl) / 3600.0
        self.e_min = c.e_reserve_wh + need
        # hard eclipse-survival floor: energy needed to carry the idle load to eclipse exit
        surv_ecl = c.p_idle_w * e.ecl_remaining_s / c.eta_discharge
        surv_sun = np.maximum(0.0, c.p_idle_w * e.next_ecl_dur_s / c.eta_discharge
                              - max(0.0, p_sun - c.p_idle_w) * c.eta_charge * e.t_to_eclipse_s)
        self.e_survive = c.e_crit_wh + np.where(e.sunlit, surv_sun, surv_ecl) / 3600.0
        self.e_survive_static = np.full_like(self.e_survive, c.e_crit_wh)
        max_ecl = float(e.ecl_remaining_s.max())
        self.e_min_static = np.full_like(self.e_min, c.e_reserve_wh + p_svc * max_ecl / c.eta_discharge / 3600.0)

    # --- helpers -------------------------------------------------------------
    def best_isl_side(self) -> Optional[str]:
        c, k = self.cfg, self.k
        if not c.isl_enabled or self.shed:
            return None
        if self.eph.sunlit[k]:          # swarm offload only during local eclipse
            return None
        sides = [s for s in ("lead", "trail") if self.node.isl_available(k, s)]
        if not sides:
            return None
        return max(sides, key=lambda s: self.nb_bucket[s])

    def _log(self, action: str, task: Task, detail: str = "", score: float = float("nan")):
        if len(self.events) < self.max_events:
            self.events.append(dict(t_h=self.t / 3600.0, task_id=task.task_id, action=action,
                                    kind=task.kind, scene=task.scene, entropy=round(task.entropy, 3),
                                    priority=task.priority, size_mb=round(task.size_mb, 2),
                                    w_est=round(task.est_value, 4), score=score,
                                    lat=round(task.lat, 2), lon=round(task.lon, 2),
                                    soc=round(self.node.E / self.cfg.battery_wh * 100, 1),
                                    temp_c=round(self.node.T - KELVIN, 2), detail=detail))

    def _deliver(self, task: Task, t_del: float, route: str):
        f = task.freshness(t_del)
        val = task.true_value * f
        self.cnt["delivered"] += 1
        if task.priority == 4 and f > 0:
            self.cnt["alerts_ok"] += 1
        self.aoi_u = max(self.aoi_u, task.t_gen)
        self.deliveries.append(dict(t_h=t_del / 3600.0, task_id=task.task_id, route=route,
                                    scene=task.scene, priority=task.priority, value=val,
                                    latency_min=(t_del - task.t_gen) / 60.0))

    def _start(self, res: str, task: Task, meta: dict):
        if task.kind == "raw":
            self.raw_q.remove(task)
        else:
            self.prod_q.remove(task)
        self.jobs[res] = task
        self.job_meta[res] = meta
        if res == "gpu":
            task.remaining = task.gflops
        else:
            task.remaining = task.size_mb
        if res == "isl":
            self.nb_bucket[meta["side"]] -= task.gflops
        self._log(RESOURCE_ACTION[res], task, meta.get("side") or "", meta.get("phi", float("nan")))

    def _complete(self, res: str, task: Task, t_c: float):
        c = self.cfg
        if res == "gpu":
            self.cnt["processed"] += 1
            prod = replace(task, size_mb=task.size_mb * c.product_ratio, gflops=0.0,
                           true_value=task.true_value * c.eta_process,
                           est_value=task.est_value * c.eta_process, kind="product", parent_id=task.task_id,
                           remaining=0.0)
            self.prod_q.append(prod)
        elif res == "dl":
            if task.kind == "raw":
                self.cnt["tx_raw"] += 1
            else:
                self.cnt["tx_prod"] += 1
            self._deliver(task, t_c, "downlink-" + task.kind)
        else:  # isl
            self.cnt["offloaded"] += 1
            side = self.job_meta["isl"]["side"]
            t_p = t_c + task.gflops / c.nb_spare_gflops
            kk = min(int(t_p / c.dt), len(self.eph.t) - 1)
            t_del = t_p + self.eph.nb_wait_contact[side][kk]
            prod = replace(task, kind="product", true_value=task.true_value * c.eta_process,
                           size_mb=task.size_mb * c.product_ratio)
            heapq.heappush(self.pending_nb, (t_del, task.task_id, prod))

    def _storage_used(self) -> float:
        s = sum(t.size_mb for t in self.raw_q) + sum(t.size_mb for t in self.prod_q)
        for r in ("gpu", "dl", "isl"):
            j = self.jobs[r]
            if j is not None:
                s += j.size_mb
        return s

    # --- main loop -----------------------------------------------------------
    def run(self, progress=None) -> SimResult:
        c, e, node = self.cfg, self.eph, self.node
        N = c.n_slots
        dt = c.dt
        tel = {name: np.zeros(N) for name in (
            "t_h", "soc", "E_wh", "E_min_wh", "temp_c", "q_mb", "m_mb", "gpu_util", "dl_mb", "isl_mb",
            "action", "sunlit", "contact", "isl_avail", "p_solar_w", "p_load_w", "theta_deg",
            "lat", "lon", "H_bar", "aoi_min", "value_cum", "energy_wh_cum", "D_E", "D_T", "shed",
            "throttled", "lyapunov")}
        value_cum, energy_cum = 0.0, 0.0
        gpu_cap = c.gpu_gflops * dt
        dl_cap = c.dl_mb_per_s * dt
        isl_cap = c.isl_mb_per_s * dt
        Tsafe = c.t_safe_c + KELVIN
        for k in range(N):
            self.k, self.t = k, k * dt
            t = self.t
            sunlit = bool(e.sunlit[k])
            # 1. neighbour deliveries that became due
            while self.pending_nb and self.pending_nb[0][0] <= t:
                t_del, _, prod = heapq.heappop(self.pending_nb)
                self._deliver(prod, t_del, "isl-neighbour")
            # 2. arrivals with ring-buffer overflow protection
            for task in self.arrivals[k] if k < len(self.arrivals) else []:
                self.cnt["generated"] += 1
                if task.priority == 4:
                    self.cnt["alerts"] += 1
                self.raw_q.append(task)
                self._log("STORE", task, "arrival")
            used = self._storage_used()
            while used > c.storage_mb and (self.raw_q or self.prod_q):
                victim = self.raw_q.pop(0) if self.raw_q else self.prod_q.pop(0)
                used -= victim.size_mb
                self.cnt["overflow"] += 1
                self.value_lost_overflow += victim.true_value
            # 3. AoI expiry
            if k % 6 == 0:
                keep_r = [x for x in self.raw_q if t - x.t_gen <= x.aoi_deadline_s]
                keep_p = [x for x in self.prod_q if t - x.t_gen <= x.aoi_deadline_s]
                self.cnt["expired"] += (len(self.raw_q) - len(keep_r)) + (len(self.prod_q) - len(keep_p))
                self.raw_q, self.prod_q = keep_r, keep_p
            # 4. hardware interlocks (identical for every controller)
            if self.shed:
                self.shed = node.E < c.e_crit_wh + 5.0
            elif node.E < c.e_crit_wh:
                self.shed = True
            if self.throttled:
                self.throttled = node.T > c.t_max_c + KELVIN - 5.0
            elif node.T >= c.t_max_c + KELVIN:
                self.throttled = True
            # neighbour spare-compute buckets refill while the donor is sunlit
            for side in ("lead", "trail"):
                if e.nb_sunlit[side][k]:
                    self.nb_bucket[side] = min(c.nb_bucket_gflop, self.nb_bucket[side] + c.nb_spare_gflops * dt)
            isl_ok = self.best_isl_side() is not None
            budget = {"gpu": gpu_cap if not (self.shed or self.throttled) else 0.0,
                      "dl": dl_cap if (e.contact[k] and not self.shed) else 0.0,
                      "isl": isl_cap if isl_ok else 0.0}
            spent = {"gpu": 0.0, "dl": 0.0, "isl": 0.0}
            self.backlog_mb = sum(x.size_mb for x in self.raw_q) + sum(x.size_mb for x in self.prod_q)
            # 5. service + decisions
            for _ in range(40):
                for res in ("gpu", "dl", "isl"):
                    job = self.jobs[res]
                    if job is not None and budget[res] > 1e-9:
                        use = min(budget[res], job.remaining)
                        job.remaining -= use
                        budget[res] -= use
                        spent[res] += use
                        if job.remaining <= 1e-9:
                            cap = {"gpu": gpu_cap, "dl": dl_cap, "isl": isl_cap}[res]
                            self._complete(res, job, t + dt * spent[res] / cap)
                            self.jobs[res] = None
                free = [r for r in ("gpu", "dl", "isl") if self.jobs[r] is None and budget[r] > 1e-6]
                if not free or not (self.raw_q or self.prod_q):
                    break
                assigned = self.policy.decide(self, free)
                if not assigned:
                    break
                for res, task, meta in assigned:
                    if res in free and self.jobs[res] is None:
                        self._start(res, task, meta)
                self.backlog_mb = sum(x.size_mb for x in self.raw_q) + sum(x.size_mb for x in self.prod_q)
            # 6. power and physics (RK4)
            f_gpu = spent["gpu"] / gpu_cap
            f_dl = spent["dl"] / dl_cap
            f_isl = spent["isl"] / isl_cap
            if self.shed:
                p_load = c.p_bus_w
            else:
                p_load = (c.p_bus_w + c.p_gpu_idle_w * (1 - f_gpu) + c.p_gpu_active_w * f_gpu
                          + c.p_tx_w * f_dl + c.p_isl_w * f_isl)
            node.step(k, p_load, dt)
            energy_cum += p_load * dt / 3600.0
            # 7. telemetry
            if self.deliveries:
                value_cum = value_cum  # updated below from new deliveries
            action = ("OFFLOAD_ISLL" if f_isl > 0 else "TRANSMIT" if f_dl > 0
                      else "PROCESS" if f_gpu > 0 else "STORE")
            q_mb = sum(x.size_mb for x in self.raw_q) + sum(x.size_mb for x in self.prod_q)
            D_E = max(0.0, self.e_min[k] - node.E)
            D_T = max(0.0, node.T - Tsafe)
            tel["t_h"][k] = (t + dt) / 3600.0
            tel["soc"][k] = node.E / c.battery_wh * 100.0
            tel["E_wh"][k] = node.E
            tel["E_min_wh"][k] = self.e_min[k]
            tel["temp_c"][k] = node.T - KELVIN
            tel["q_mb"][k] = q_mb
            tel["m_mb"][k] = self._storage_used()
            tel["gpu_util"][k] = f_gpu
            tel["dl_mb"][k] = spent["dl"]
            tel["isl_mb"][k] = spent["isl"]
            tel["action"][k] = ACTION_CODE[action]
            tel["sunlit"][k] = sunlit
            tel["contact"][k] = e.contact[k]
            tel["isl_avail"][k] = isl_ok
            tel["p_solar_w"][k] = node.p_solar_series[k]
            tel["p_load_w"][k] = p_load
            tel["theta_deg"][k] = e.theta_deg[k]
            tel["lat"][k] = e.lat[k]
            tel["lon"][k] = e.lon[k]
            tel["H_bar"][k] = (np.mean([x.entropy for x in self.raw_q]) if self.raw_q else 0.0)
            tel["aoi_min"][k] = (t + dt - self.aoi_u) / 60.0
            tel["energy_wh_cum"][k] = energy_cum
            tel["D_E"][k] = D_E
            tel["D_T"][k] = D_T
            tel["shed"][k] = self.shed
            tel["throttled"][k] = self.throttled
            tel["lyapunov"][k] = 0.5 * ((q_mb / c.storage_mb) ** 2 + (D_E / 5.0) ** 2 + (D_T / 5.0) ** 2)
            if progress is not None and k % 500 == 0:
                progress(k / N)
        # cumulative delivered value on the telemetry grid
        if self.deliveries:
            dt_h = np.array([d["t_h"] for d in self.deliveries])
            dv = np.array([d["value"] for d in self.deliveries])
            order = np.argsort(dt_h)
            cum = np.cumsum(dv[order])
            pos = np.searchsorted(dt_h[order], tel["t_h"], side="right")
            tel["value_cum"] = np.where(pos > 0, cum[np.maximum(pos - 1, 0)], 0.0)
        return SimResult(self.policy.name, tel, self.events, self._metrics(tel), self.deliveries)

    def _metrics(self, tel) -> Dict[str, float]:
        c = self.cfg
        gen_value = sum(t.true_value for slot in self.arrivals[:c.n_slots] for t in slot)
        # contact-limited ceiling: every task delivered at the first ground contact after
        # generation with zero processing/transfer time and unlimited resources
        ceiling = 0.0
        for kk, slot in enumerate(self.arrivals[:c.n_slots]):
            w = self.eph.wait_contact[kk]
            for t in slot:
                if kk * c.dt + w < c.n_slots * c.dt:
                    ceiling += t.true_value * t.freshness(t.t_gen + w)
        val = sum(d["value"] for d in self.deliveries)
        lat = np.array([d["latency_min"] for d in self.deliveries]) if self.deliveries else np.array([np.nan])
        dur_s = c.n_slots * c.dt
        contact_s = float(tel["contact"].sum() * c.dt)
        payload_wh = float(np.sum(tel["p_load_w"] - c.p_bus_w) * c.dt / 3600.0)
        Tsafe = c.t_safe_c
        return dict(
            value=val,
            value_capture_pct=100.0 * val / gen_value if gen_value else 0.0,
            ceiling_value=ceiling,
            ceiling_pct=100.0 * val / ceiling if ceiling else 0.0,
            generated=self.cnt["generated"],
            delivered=self.cnt["delivered"],
            completion_pct=100.0 * self.cnt["delivered"] / max(1, self.cnt["generated"]),
            alerts=self.cnt["alerts"],
            alerts_on_time_pct=100.0 * self.cnt["alerts_ok"] / max(1, self.cnt["alerts"]),
            processed=self.cnt["processed"],
            offloaded=self.cnt["offloaded"],
            tx_raw=self.cnt["tx_raw"],
            tx_prod=self.cnt["tx_prod"],
            overflow=self.cnt["overflow"],
            expired=self.cnt["expired"],
            energy_total_wh=float(tel["energy_wh_cum"][-1]),
            energy_payload_wh=payload_wh,
            value_per_wh=val / max(1e-9, payload_wh),
            latency_mean_min=float(np.nanmean(lat)),
            latency_p95_min=float(np.nanpercentile(lat, 95)) if self.deliveries else float("nan"),
            aoi_mean_min=float(np.mean(tel["aoi_min"])),
            soc_min_pct=float(tel["soc"].min()),
            soc_mean_pct=float(tel["soc"].mean()),
            shed_pct=100.0 * float(tel["shed"].mean()),
            below_emin_pct=100.0 * float((tel["E_wh"] < tel["E_min_wh"]).mean()),
            temp_max_c=float(tel["temp_c"].max()),
            above_tsafe_pct=100.0 * float((tel["temp_c"] > Tsafe).mean()),
            throttled_pct=100.0 * float(tel["throttled"].mean()),
            gpu_duty_pct=100.0 * float(tel["gpu_util"].mean()),
            dl_util_pct=100.0 * float(tel["dl_mb"].sum()) / max(1e-9, contact_s * c.dl_mb_per_s),
            isl_gb=float(tel["isl_mb"].sum()) / 1000.0,
            contact_min_per_day=contact_s / 60.0 / (dur_s / 86400.0),
            backlog_mean_mb=float(tel["q_mb"].mean()),
        )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------
def make_policies(V: float = 0.3, include_ablations: bool = False) -> List[Policy]:
    pols: List[Policy] = [ELotusEngine(V=V), AlwaysTransmit(), GreedyProcess(), StoreAndForward(), RuleBased()]
    if include_ablations:
        pols += [ELotusEngine(V=V, use_isl=False, label="E-LOTUS w/o ISLL"),
                 ELotusEngine(V=V, use_entropy=False, label="E-LOTUS w/o entropy"),
                 ELotusEngine(V=V, eclipse_aware=False, label="E-LOTUS static E_min"),
                 ELotusEngine(V=V, estimator="entropy", label="E-LOTUS raw entropy")]
    return pols


def run_simulation(cfg: SimConfig, policy: Policy, eph: Optional[Ephemeris] = None,
                   arrivals=None, progress=None) -> SimResult:
    """24-hour (cfg.duration_h) simulation loop returning full time-series logs."""
    eph = eph if eph is not None else build_ephemeris(cfg)
    arrivals = arrivals if arrivals is not None else generate_workload(cfg, eph)
    return Simulation(cfg, policy, eph, arrivals).run(progress)


def run_comparison(cfg: SimConfig, policies: List[Policy]) -> Dict[str, SimResult]:
    eph = build_ephemeris(cfg)
    arrivals = generate_workload(cfg, eph)
    return {p.name: Simulation(cfg, p, eph, arrivals).run() for p in policies}


if __name__ == "__main__":
    import time
    cfg = SimConfig()
    t0 = time.time()
    eph = build_ephemeris(cfg)
    arr = generate_workload(cfg, eph)
    print(f"ephemeris+workload {time.time()-t0:.2f}s | contact {eph.contact[:cfg.n_slots].mean()*1440:.1f} min/day"
          f" | sunlit {eph.sunlit[:cfg.n_slots].mean()*100:.1f}% | beta {eph.beta_deg[0]:.1f} deg")
    for p in make_policies(include_ablations=True):
        t0 = time.time()
        r = Simulation(cfg, p, eph, arr).run()
        m = r.metrics
        print(f"{p.name:22s} {time.time()-t0:5.1f}s value={m['value']:7.1f} cap={m['value_capture_pct']:5.1f}% "
              f"compl={m['completion_pct']:5.1f}% alerts={m['alerts_on_time_pct']:5.1f}% "
              f"E={m['energy_payload_wh']:6.1f}Wh socmin={m['soc_min_pct']:5.1f} Tmax={m['temp_max_c']:5.1f} "
              f"lat={m['latency_mean_min']:6.1f} ovf={m['overflow']} exp={m['expired']} shed={m['shed_pct']:.1f}% thr={m['throttled_pct']:.1f}%")
