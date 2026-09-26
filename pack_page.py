"""Pack web/orbitra_template.html + web/orbitra_data.json + Earth textures into one self-contained page.
Run after build_web_data.py:   python web/pack_page.py
"""
import base64
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engine import SimConfig, build_ephemeris  # noqa: E402

D = json.load(open(os.path.join(HERE, "orbitra_data.json")))
G = D["grid"]


def fk(x):
    return str(x)


rep = {}
for key, v in D["replay"].items():
    parts = key.split("|")
    if parts[0] == "E-LOTUS":
        _, V, s, k, st, i = parts
        rep["e_%d_%d_%d_%d_%d" % (G["V"].index(float(V)), G["soc"].index(int(s)), G["k"].index(float(k)),
                                  G["st"].index(int(st)), G["isl"].index(int(i)))] = v
    else:
        p, s, k, st, i = parts
        tag = "gp" if p == "Greedy-Process" else "rb"
        rep["%s_%d_%d_%d_%d" % (tag, G["soc"].index(int(s)), G["k"].index(float(k)),
                                G["st"].index(int(st)), G["isl"].index(int(i)))] = v
orb = {}
for key, v in D["orbits"].items():
    if key == "real":
        orb["real"] = v
    else:
        inc, lt = key.split("|")
        orb["o_%d_%d" % (G["inc"].index(float(inc)), G["ltan"].index(float(lt)))] = v
D["replay"], D["orbits"] = rep, orb

# real orbit plane: inclination and ascending-node local time
c = SimConfig()
e = build_ephemeris(c, extra_h=0.0)
h = np.cross(e.r[0], e.v[0])
node = np.cross([0.0, 0.0, 1.0], h)
raan = math.degrees(math.atan2(node[1], node[0]))
ra_sun = math.degrees(math.atan2(e.sun[0, 1], e.sun[0, 0]))
D["real_orbit"] = dict(inc=round(math.degrees(math.acos(h[2] / np.linalg.norm(h))), 2),
                       ltan=round(((raan - ra_sun) / 15.0 + 12.0) % 24.0, 3), alt=G["alt"])

blob = base64.b64encode(gzip.compress(json.dumps(D, separators=(",", ":")).encode(), 9)).decode()


def uri(name, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode(open(os.path.join(HERE, "assets", name), "rb").read()).decode()


html = open(os.path.join(HERE, "orbitra_template.html")).read()
html = (html.replace("__DATA__", blob)
            .replace("__EARTH_DAY__", uri("earth_day_2k.jpg"))
            .replace("__EARTH_NIGHT__", uri("earth_night_2k.jpg"))
            .replace("__EARTH_MAP__", uri("earth_day_1k.jpg")))
out = os.path.join(HERE, "orbitra_mission_control.html")
open(out, "w").write(html)
print(f"wrote {out}: {os.path.getsize(out) / 1e6:.2f} MB (data {len(blob) / 1e6:.2f} MB)")
