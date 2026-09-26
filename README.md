# E-LOTUS: IASTAM 6.0 Phase 2 submission bundle (real-data edition)
**E**ntropy-Weighted **L**yapunov **O**ptimization with **T**hermal Radiation & **S**warm Offloading (E-LOTUS)
Track 1, Problem 1: "Process or Transmit?" | Theme: *Data Centers Beyond Earth*

## What is real
| Input | Source |
|---|---|
| Orbit | **Φsat-2** (ESA onboard-AI satellite, NORAD 60470): CelesTrak TLE, epoch 2026-09-26, propagated with **SGP4** (`data/phisat2_tle.txt`). An independent RK4 two-body+J2 integrator agrees within ~1.6 km over 24 h. |
| Optical quick-looks | NASA GIBS `VIIRS_NOAA20_CorrectedReflectance_TrueColor`, daily, sampled along the real ground track |
| Thermal-IR quick-looks (eclipse) | NASA GIBS `VIIRS_NOAA20_Brightness_Temp_BandI5_Night` |
| Fire alerts | NASA FIRMS **VIIRS 375 m active fires** (`VIIRS_NOAA20_Thermal_Anomalies_375m_All`) |
| Cloud fraction (ground truth) | **MODIS Aqua** cloud fraction (day/night), decoded with the GIBS colour map |
| Land fraction | OpenStreetMap land/water map (GIBS `OSM_Land_Water_Map`) |
| Ground stations | KSAT SvalSat (78.23 N, 15.40 E) and ASI Matera (40.65 N, 16.70 E) |
| Window | 2026-08-26 to 2026-09-14 (20 days, 57600 along-track samples every 30 s) |

`data/elotus_gibs_atlas.bin` holds the sampled real-data atlas. `data/gibs_sampler.js` regenerates it: run it in a browser console on https://gibs.earthdata.nasa.gov/.

**What remains modelled.** Acquisition timing (Poisson strip mapping), scene size and GFLOP cost, the power/thermal/compute hardware, and the neighbour-donor model are simulated. The paper's Limitations section lists all of these.

## Key real-data finding
Raw quick-look entropy **misranks clouds**. Clouds make up 64% of scenes and are the most textured, with a median H of 6.9 bit/px against 6.3 for clear land. On the held-out days, the top 30% of scenes ranked by raw entropy hold only 27% of the value, worse than random (30%). The E-LOTUS estimator fuses entropy with the footprint's onboard land-mask prior and reaches 79%, against 82% for a label-aware oracle.

## Dashboards (product name: Orbitra, powered by the E-LOTUS engine)
* `app.py`: Streamlit **Orbitra Mission Control**. It holds the live what-if simulation (V, SOC, storage, radiator, ISL, real day) and a 3D live orbit over a NASA Blue Marble Earth with the real day/night terminator. The **Orbit designer** covers inclination and ascending-node local time, with battery, eclipse and contact maps plus a "Find the best orbit" search over 40 full engine runs.
* `web/orbitra_mission_control.html`: standalone presentation page. It opens on an interactive 3D Earth where you drag the satellite along its orbit and pick any of 80 pre-computed what-if orbits. It also has a 270-run what-if replay with 0.5×/1×/2× speed (1× plays 24 h in 3 min). Rebuild it with `python web/build_web_data.py && python web/pack_page.py`.
* What-if orbits use the RK4 two-body + J2 integrator at the real mean altitude. Each footprint takes the nearest real NASA atlas sample (same day preferred).

## Quick start
```bash
pip install -r requirements.txt
python -m streamlit run app.py    # Orbitra dashboard: what-if simulation, orbit designer, telemetry, Pareto, live decision feed
python engine.py                  # 24-h comparison on the first real day
python benchmark.py               # 20-day real-data suite that regenerates every number and figure (about 6 min on 2 cores)
python make_paper_numbers.py      # results -> paper/numbers.tex + tables
cd paper && pdflatex elotus.tex && pdflatex elotus.tex
```

## Headline results (20 real days, seed 2026, per-day averages)
| Controller | Value/day | % of contact-limited ceiling | Completion | Payload energy | Load-shed |
|---|---|---|---|---|---|
| **E-LOTUS** | 152.1 | 86.2% | 83.2% | 217.1 Wh | 0.0% |
| Rule-Based | 144.0 | 81.6% | 49.7% | 160.7 Wh | 0.0% |
| Store-and-Forward | 72.7 | 41.2% | 7.5% | 51.8 Wh | 0.0% |
| Always-Transmit | 3.4 | 1.9% | 7.4% | 51.8 Wh | 0.0% |
| Greedy-Process | 5.7 | 3.2% | 10.1% | 218.6 Wh | 39.0% |

E-LOTUS delivered more value than Rule-Based in all five stress scenarios, each run on different real days.

## Bundle map
| Path | Contents |
|---|---|
| `paper/elotus.pdf` + `.tex` | Interim research paper (IEEE format). Every number is an auto-generated macro. |
| `pitch/pitch_script.md` | Timed 2-minute video script |
| `engine.py` | SGP4/RK4 orbit, real-data atlas, RK4 battery and thermal models, E-LOTUS, baselines |
| `app.py` | Streamlit Orbitra mission-control dashboard (what-if simulation + orbit designer) |
| `web/` | Standalone Orbitra page, its scenario builder and packer, and the Earth textures (NASA Blue Marble / Black Marble, public domain) |
| `benchmark.py`, `make_paper_numbers.py` | Reproduce every table and figure |
| `data/` | Real TLE, NASA GIBS atlas, sampler script, ground-track file |
| `results/` | `summary.json`, `realdata_stats.json`, CSVs |
| `diagrams/` | PlantUML sources and PNGs (Figs. 1–2) |

The compute module is modelled as a "Jetson Orin-class module, 15 W mode". NVIDIA does not sell a product named "Jetson Orin Space Module".
