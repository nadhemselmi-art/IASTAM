"""Real ground track of PHISAT-2 (NORAD 60470) from its CelesTrak TLE, propagated with SGP4.
Writes track_30s.npz and base64 chunks used by the in-browser NASA GIBS sampler."""
import numpy as np, base64, json, datetime as dt, math, sys
sys.path.insert(0, '/home/claude/elotus')
from sgp4.api import Satrec, jday
from engine import sun_unit_vector, gmst_rad, R_EARTH
L = open('/home/claude/elotus/data/phisat2_tle.txt').read().splitlines()
sat = Satrec.twoline2rv(L[1], L[2])
t0 = dt.datetime(2026, 8, 26, 0, 0, 0)
days, step = 30, 30.0
n = int(days * 86400 / step)
t = np.arange(n) * step
jd0, fr0 = jday(t0.year, t0.month, t0.day, 0, 0, 0)
fr = fr0 + t / 86400.0
jd = np.full(n, jd0)
e, r, v = sat.sgp4_array(jd, fr)
assert (e == 0).all(), "SGP4 error"
jdf = jd + fr
sun = sun_unit_vector(jdf)
proj = np.einsum('ij,ij->i', r, sun)
perp = r - proj[:, None] * sun
sunlit = ~((proj < 0) & (np.linalg.norm(perp, axis=1) < R_EARTH))
gm = gmst_rad(jdf)
rn = np.linalg.norm(r, axis=1)
lat = np.degrees(np.arcsin(r[:, 2] / rn))
lon = (np.degrees(np.arctan2(r[:, 1], r[:, 0]) - gm) + 180) % 360 - 180
W, H = 4096, 2048
x = np.clip(np.floor((lon + 180) / 360 * W), 0, W - 1).astype(np.uint32)
y = np.clip(np.floor((90 - lat) / 180 * H), 0, H - 1).astype(np.uint32)
packed = (x << 12) | (y << 1) | (~sunlit).astype(np.uint32)   # 24-bit: x(12) y(11) night(1)
b = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1).astype(np.uint8).ravel()
np.savez('/home/claude/elotus/data/track_30s.npz', t=t, lat=lat, lon=lon, sunlit=sunlit, alt=rn - R_EARTH, x=x, y=y)
per_day = int(86400 / step)
chunks = []
for d in range(0, days, 6):
    seg = b[d * per_day * 3:(d + 6) * per_day * 3]
    chunks.append(base64.b64encode(seg.tobytes()).decode())
json.dump(chunks, open('/home/claude/elotus/data/track_chunks.json', 'w'))
print('positions', n, 'sunlit frac', sunlit.mean().round(3), 'alt range', (rn - R_EARTH).min().round(1), (rn - R_EARTH).max().round(1))
print('chunk sizes', [len(c) for c in chunks])
