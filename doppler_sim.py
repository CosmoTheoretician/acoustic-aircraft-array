#!/usr/bin/env python3
"""
One-way S-band Doppler velocimetry of lunar spacecraft from the garden dish: profiles and
error budget.

Geometry is computed locally (no site coordinates leave this machine):
  - station: astropy EarthLocation -> GCRS (full IAU precession-nutation and Earth rotation),
  - Moon: astropy built-in lunar theory (km-level, plenty for profiles and error budgets;
    for real data, use JPL Horizons *geocentric* vectors and this same station code),
  - lunar orientation: IAU/WGCCRE 2009 rotation model (alpha0, delta0, W with E1..E13),
  - lander at Reiner Gamma (7.5 N, 59.0 W), and a generic 100 km polar lunar orbiter.

Measured line-of-sight velocity (one-way, in m/s):
  v_meas = rhodot + c (y_rx - y_sc - y_gr) + d(trop)/dt - d(iono)/dt + noise(C/N0)
  y_rx   receiver reference (GPSDO) fractional frequency error,
  y_sc   spacecraft oscillator error (unknown offset + drift + wander in one-way mode),
  y_gr   gravitational redshift + 2nd-order Doppler (nearly constant, ends up in the bias),
  trop   zenith delay 2.3 m mapped with 1/sin(el); iono: diurnal TEC, single-layer mapping.
Per pass, the model (ephemeris with station / lander position errors, troposphere and
ionosphere models with given errors) is subtracted and a bias + drift is fitted: that is
what one-way Doppler can deliver without knowing the spacecraft oscillator.

Run:  .venv/bin/python doppler_sim.py [--site site_local.json]
Writes out/dish/doppler_* (example site) or out_local/dish/ (--site).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time

import dish_params as DP
from tracking_sim import example_site

C = DP.C
R_MOON = 1737.4e3
HERE = os.path.dirname(os.path.abspath(__file__))
LANDER_LAT, LANDER_LON = 7.5, -59.0  # Reiner Gamma (selenographic)


# ---------------------------------------------------------------------------- geometry
def moon_rotation(t: Time):
    """IAU/WGCCRE 2009 Moon orientation: matrices body-fixed -> ICRF/GCRS, shape (N,3,3)."""
    d = (t.tdb.jd - 2451545.0)
    T = d / 36525
    E = np.radians(np.array([
        125.045 - 0.0529921 * d, 250.089 - 0.1059842 * d, 260.008 + 13.0120009 * d,
        176.625 + 13.3407154 * d, 357.529 + 0.9856003 * d, 311.589 + 26.4057084 * d,
        134.963 + 13.0649930 * d, 276.617 + 0.3287146 * d, 34.226 + 1.7484877 * d,
        15.134 - 0.1589763 * d, 119.743 + 0.0036096 * d, 239.961 + 0.1643573 * d,
        25.053 + 12.9590088 * d]))
    s, c = np.sin(E), np.cos(E)
    a0 = (269.9949 + 0.0031 * T - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2] - 0.0172 * s[3]
          + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12])
    d0 = (66.5392 + 0.0130 * T + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2] + 0.0068 * c[3]
          - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9] - 0.0009 * c[12])
    W = (38.3213 + 13.17635815 * d - 1.4e-12 * d ** 2 + 3.5610 * s[0] + 0.1208 * s[1] - 0.0642 * s[2]
         + 0.0158 * s[3] + 0.0252 * s[4] - 0.0066 * s[5] - 0.0047 * s[6] - 0.0046 * s[7] + 0.0028 * s[8]
         + 0.0052 * s[9] + 0.0040 * s[10] + 0.0019 * s[11] - 0.0044 * s[12])
    a, dd, w = np.radians(a0) + np.pi / 2, np.pi / 2 - np.radians(d0), np.radians(W)

    def rz(x):
        cx, sx = np.cos(x), np.sin(x)
        z, o = np.zeros_like(x), np.ones_like(x)
        return np.stack([np.stack([cx, -sx, z], -1), np.stack([sx, cx, z], -1), np.stack([z, z, o], -1)], -2)

    def rx(x):
        cx, sx = np.cos(x), np.sin(x)
        z, o = np.zeros_like(x), np.ones_like(x)
        return np.stack([np.stack([o, z, z], -1), np.stack([z, cx, -sx], -1), np.stack([z, sx, cx], -1)], -2)

    return rz(a) @ rx(dd) @ rz(w)


def moon_pos(t):
    m = get_body("moon", t, ephemeris="builtin")
    return m.cartesian.xyz.to(u.m).value.T


ORBIT_EPOCH = Time("2027-03-15T00:00:00", scale="utc")


def target_pos(kind, t, orbit_phase0=0.0, lander=(LANDER_LAT, LANDER_LON)):
    rm = moon_pos(t)
    if kind == "lander":
        la, lo = np.radians(lander[0]), np.radians(lander[1])
        rb = R_MOON * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        return rm + moon_rotation(t) @ rb
    # 100 km circular polar orbit; plane fixed in inertial space (contains the lunar pole)
    a = R_MOON + 100e3
    n = np.sqrt(4.9028e12 / a ** 3)
    R0 = moon_rotation(ORBIT_EPOCH.reshape((1,)))[0]
    pole, x0 = R0[:, 2], R0[:, 0]
    ph = orbit_phase0 + n * (t - ORBIT_EPOCH).to(u.s).value
    return rm + a * (np.cos(ph)[:, None] * x0 + np.sin(ph)[:, None] * pole)


def target_posvel(kind, t, **kw):
    h = 0.5 * u.s
    rp, rmn = target_pos(kind, t + h, **kw), target_pos(kind, t - h, **kw)
    return (rp + rmn) / 2, (rp - rmn) / 1.0


def station(site, t):
    loc = EarthLocation.from_geodetic(site["lon"] * u.deg, site["lat"] * u.deg, site["ground_elev_m"] * u.m)
    pos, vel = loc.get_gcrs_posvel(t)
    return pos.xyz.to(u.m).value.T, vel.xyz.to(u.m / u.s).value.T


def perturb_site(site, rng, err_m):
    """Station error is Earth-fixed: random 3D offset of err_m applied to lat/lon/height."""
    d = rng.normal(size=3)
    d *= err_m / np.linalg.norm(d)
    s = dict(site)
    s["lat"] = site["lat"] + d[1] / 111_320
    s["lon"] = site["lon"] + d[0] / (111_320 * np.cos(np.radians(site["lat"])))
    s["ground_elev_m"] = site["ground_elev_m"] + d[2]
    return s


def perturb_lander(rng, err_m):
    """Lander position error is Moon-fixed: random horizontal offset of err_m."""
    a = rng.uniform(0, 2 * np.pi)
    r = err_m / R_MOON
    return (LANDER_LAT + np.degrees(r * np.cos(a)),
            LANDER_LON + np.degrees(r * np.sin(a)) / np.cos(np.radians(LANDER_LAT)))


def geometry(kind, site, t, **kw):
    """Range, range-rate (with light time), elevation, occultation flag."""
    rs, vs = station(site, t)
    rt, vt = target_posvel(kind, t, **kw)
    rho = np.linalg.norm(rt - rs, axis=1)
    rt = rt - vt * (rho / C)[:, None]  # light time, first order
    los = rt - rs
    rho = np.linalg.norm(los, axis=1)
    uv = los / rho[:, None]
    rdot = np.einsum("ij,ij->i", uv, vt - vs)
    up = rs / np.linalg.norm(rs, axis=1)[:, None]
    el = np.degrees(np.arcsin(np.einsum("ij,ij->i", uv, up)))
    occ = np.zeros(len(t), bool)
    if kind == "orbiter":
        rm = moon_pos(t)
        s = np.einsum("ij,ij->i", rm - rs, uv)
        closest = np.linalg.norm(rs + uv * s[:, None] - rm, axis=1)
        occ = (closest < R_MOON) & (s < rho)
    return rho, rdot, el, occ, rs


def iono_trop(el, t, site, f_hz):
    """Slant tropospheric delay (m, +) and ionospheric phase advance (m, +), simple models."""
    sel = np.sin(np.radians(np.clip(el, 3, 90)))
    trop = 2.3 / sel
    lst = (t.utc.mjd % 1.0) * 24 + site["lon"] / 15  # local mean solar time, h
    vtec = np.clip(12 + 9 * np.sin(2 * np.pi * (lst - 8) / 24), 3, None)  # TECU
    re, hi = 6371e3, 350e3
    mf = 1 / np.sqrt(1 - (re * np.cos(np.radians(el)) / (re + hi)) ** 2)
    iono = 40.3 * vtec * mf * 1e16 / f_hz ** 2
    return trop, iono


# ---------------------------------------------------------------------------- noise
def sigma_v(cn0_dbhz, T, lam):
    return lam * np.sqrt(3 / (np.pi ** 2 * 10 ** (cn0_dbhz / 10) * T ** 3))


def iq_check(rng, cn0s, lam):
    """Generate the carrier at baseband (after removing a predicted Doppler), estimate the
    frequency every 1 s with a zero-padded FFT + parabolic peak, compare with the CRLB."""
    fs, Tn, rows = 400.0, 1.0, []
    n = int(fs * Tn)
    t = np.arange(int(300 * fs)) / fs
    f_res = 3.0 + 0.002 * t  # residual Doppler after prediction: a few Hz, slowly drifting
    phase = 2 * np.pi * np.cumsum(f_res) / fs
    for cn0 in cn0s:
        a = np.sqrt(10 ** (cn0 / 10) / fs)  # unit-noise power per sample -> C/N0
        x = a * np.exp(1j * phase) + (rng.normal(size=t.size) + 1j * rng.normal(size=t.size)) / np.sqrt(2)
        est, tru = [], []
        for k in range(t.size // n):
            seg = x[k * n:(k + 1) * n]
            N = 16 * n
            X = np.abs(np.fft.fft(seg * np.hanning(n), N))
            i = int(np.argmax(X))
            y0, y1, y2 = X[i - 1], X[i], X[(i + 1) % N]
            di = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            fk = (i + di) * fs / N
            est.append(fk - fs if fk > fs / 2 else fk)
            tru.append(f_res[k * n:(k + 1) * n].mean())
        err = np.array(est) - np.array(tru)
        rows.append((cn0, np.std(err) * lam * 1e3, sigma_v(cn0, Tn, lam) * 1e3))
    return rows


def osc(rng, n, white_1s, rw_per_sqrt_s, offset, drift_per_day):
    """Fractional frequency series at 1 s: offset + linear drift + random walk FM + white FM."""
    tt = np.arange(n)
    return (offset + drift_per_day * tt / 86400 + np.cumsum(rng.normal(0, rw_per_sqrt_s, n))
            + rng.normal(0, white_1s, n))


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="JSON with lat, lon, ground_elev_m (git-ignored for your own site)")
    ap.add_argument("--start", default="2027-03-15T00:00:00")
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    if args.site:
        with open(args.site) as f:
            site = json.load(f)
        out_dir = os.path.join(HERE, "out_local", "dish")
    else:
        site = example_site()
        out_dir = os.path.join(HERE, "out", "dish")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    p = DP.DEFAULT
    f0 = p.f_s_mhz * 1e6
    lam = C / f0
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"=== One-way S-band Doppler: {site.get('name', 'site')}, {p.f_s_mhz:.1f} MHz ===")

    # ---- profiles: lander over several days, orbiter over 6 h ----
    t0 = Time(args.start, scale="utc")
    tp = t0 + np.arange(0, args.days * 86400, 60) * u.s
    rho, rdot, el, _, _ = geometry("lander", site, tp)
    vis = el > 10
    fd = -rdot / lam
    acc = np.gradient(rdot, 60.0)
    out(f"lander (Reiner Gamma): range {rho.min() / 1e3:.0f}-{rho.max() / 1e3:.0f} km; visible (el > 10 deg) "
        f"{vis.mean() * 100:.0f} % of the time; Doppler {fd[vis].min():+.0f} .. {fd[vis].max():+.0f} Hz; "
        f"rate up to {np.abs(acc[vis]).max() / lam:.3f} Hz/s")
    first = int(np.argmax(el > 20))  # start of the first good pass
    to = tp[first] + np.arange(0, 6 * 3600, 10) * u.s
    rho_o, rdot_o, el_o, occ_o, _ = geometry("orbiter", site, to)
    vo = (el_o > 10) & ~occ_o
    acc_o = np.gradient(rdot_o, 10.0)
    out(f"100 km polar orbiter: Doppler {(-rdot_o / lam)[vo].min():+.0f} .. {(-rdot_o / lam)[vo].max():+.0f} Hz, "
        f"rate up to {np.abs(acc_o[vo]).max() / lam:.1f} Hz/s; occulted {occ_o[el_o > 10].mean() * 100:.0f} % "
        f"of the visible time")

    # ---- IQ-level estimator check ----
    out("\nfrequency estimator check (300 s of simulated carrier, 1-s FFT estimates):")
    for cn0, meas, crlb in iq_check(rng, (20.0, 26.0, 41.0, 56.0), lam):
        out(f"  carrier C/N0 {cn0:4.0f} dB-Hz: measured {meas:7.2f} mm/s, CRLB {crlb:7.2f} mm/s per 1-s point")

    # ---- error budget over one pass (1-s points) ----
    vis_idx = np.where(el > 20)[0]
    # first contiguous visibility window above 20 deg
    br = np.where(np.diff(vis_idx) > 1)[0]
    seg = vis_idx[: (br[0] + 1) if len(br) else len(vis_idx)]
    ts = tp[seg[0]] + np.arange(0, (seg[-1] - seg[0]) * 60, 1.0) * u.s
    n = len(ts)
    out(f"\nerror budget: one pass above 20 deg elevation, {n / 3600:.1f} h of 1-s points, starting {ts[0].iso[:16]} UTC")
    rho_t, rdot_t, el_t, _, rs = geometry("lander", site, ts)
    trop_t, iono_t = iono_trop(el_t, ts, site, f0)
    gr = -(6.25e7 - 1.04e6 - 2.82e6) / C ** 2  # Earth-surface vs lunar-surface potential difference
    base = dict(cn0=41.2, rx_white=p.gpsdo_adev_1s, sc="TCXO", st_err=0.02, ld_err=100.0, trop_err=0.05,
                iono_err=0.20)
    SC = {"TCXO": (1e-10, 3e-13, 2e-7, 1e-8), "OCXO": (1e-12, 1e-14, 1e-8, 1e-10),
          "coherent": (1e-13, 1e-15, 1e-9, 0.0)}
    scen = [
        ("baseline: LGA upright, OCXO GPSDO 1e-11, s/c TCXO", {}),
        ("lander tipped (carrier 26 dB-Hz)", dict(cn0=26.2)),
        ("high-gain antenna (carrier 56 dB-Hz)", dict(cn0=56.1)),
        ("s/c OCXO-class oscillator", dict(sc="OCXO")),
        ("s/c coherent with an IM ground uplink", dict(sc="coherent")),
        ("s/c coherent + premium GPSDO 2e-12", dict(sc="coherent", rx_white=2e-12)),
        ("cheap GPSDO (1e-10 at 1 s)", dict(rx_white=1e-10)),
        ("coherent + premium, station from handheld GPS 5 m", dict(st_err=5.0, sc="coherent", rx_white=2e-12)),
        ("coherent + premium, landing site known to 1 km", dict(ld_err=1000.0, sc="coherent", rx_white=2e-12)),
        ("coherent + premium, no troposphere/iono model", dict(trop_err=1.0, iono_err=1.0, sc="coherent",
                                                                rx_white=2e-12)),
    ]
    out(f"{'scenario':48s} {'1-s rms':>9s} {'60-s rms':>9s} {'drift-free sys.':>15s}   (mm/s, after bias + drift fit)")
    keep = {}
    tt = np.arange(n, dtype=float)
    A = np.vstack([np.ones(n), tt / 3600]).T
    for name, ov in scen:
        s = {**base, **ov}
        white, rw, off, drift = SC[s["sc"]]
        y_sc = osc(rng, n, white, rw, rng.normal(0, off), drift)
        y_rx = osc(rng, n, s["rx_white"], s["rx_white"] * 1e-3, 0.0, 0.0)
        v_meas = (rdot_t + C * (y_rx - y_sc - gr) + np.gradient(trop_t) - np.gradient(iono_t)
                  + rng.normal(0, sigma_v(s["cn0"], 1.0, lam), n))
        _, rdot_m, _, _, _ = geometry("lander", perturb_site(site, rng, s["st_err"]), ts,
                                      lander=perturb_lander(rng, s["ld_err"]))
        model = rdot_m + np.gradient(trop_t) * (1 - s["trop_err"]) - np.gradient(iono_t) * (1 - s["iono_err"])
        res = v_meas - model
        res -= A @ np.linalg.lstsq(A, res, rcond=None)[0]
        r60 = res[: n // 60 * 60].reshape(-1, 60).mean(1)
        # systematic part: same run without thermal noise and without oscillator white noise
        sys_v = (rdot_t + np.gradient(trop_t) - np.gradient(iono_t)) - model
        sys_v -= A @ np.linalg.lstsq(A, sys_v, rcond=None)[0]
        out(f"{name:48s} {np.std(res) * 1e3:9.2f} {np.std(r60) * 1e3:9.2f} {np.std(sys_v) * 1e3:15.3f}")
        keep[name] = res
    out("  ('drift-free sys.' = ephemeris / station / lander / troposphere / ionosphere model errors only)")
    out(f"  gravitational redshift + 2nd-order Doppler: {abs(gr) * C:.2f} m/s, nearly constant -> absorbed by the bias")
    out("  one-way Doppler measures range-rate up to an unknown bias + drift per pass; with the lander's own "
        "oscillator the pass-long wander dominates unless the link is coherent with an IM uplink")

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    hrs = (tp - t0).to(u.h).value
    a0 = ax[0, 0]
    a0.plot(hrs, np.where(vis, fd / 1e3, np.nan), lw=1.2)
    a0.set_ylabel("Doppler [kHz]")
    a0.set_xlabel(f"hours from {t0.iso[:16]} UTC")
    a0.set_title("lander at Reiner Gamma: one-way Doppler (el > 10 deg)")
    a0b = a0.twinx()
    a0b.plot(hrs, np.where(vis, el, np.nan), color="C1", lw=0.8, alpha=0.6)
    a0b.set_ylabel("elevation [deg]", color="C1")
    a1 = ax[0, 1]
    ho = (to - to[0]).to(u.h).value
    a1.plot(ho, np.where(vo, -rdot_o / lam / 1e3, np.nan), lw=1.2)
    a1.set_title("100 km polar lunar orbiter: Doppler (gaps = behind the Moon or low)")
    a1.set_xlabel(f"hours from {to[0].iso[:16]} UTC")
    a1.set_ylabel("Doppler [kHz]")
    a2 = ax[1, 0]
    th = tt / 3600
    for k, col in zip(list(keep)[:3], ("C0", "C3", "C2")):
        r = keep[k]
        r60 = r[: n // 60 * 60].reshape(-1, 60).mean(1)
        a2.plot(th[: n // 60 * 60: 60], r60 * 1e3, color=col, lw=1, label=k)
    a2.set_ylabel("post-fit residual, 60-s means [mm/s]")
    a2.set_xlabel("hours into the pass")
    a2.set_title("error budget: residual after bias + drift fit")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    a3 = ax[1, 1]
    for k, col in zip(("s/c coherent with an IM ground uplink", "s/c coherent + premium GPSDO 2e-12",
                       "coherent + premium, no troposphere/iono model"), ("C4", "C5", "C6")):
        r = keep[k]
        r60 = r[: n // 60 * 60].reshape(-1, 60).mean(1)
        a3.plot(th[: n // 60 * 60: 60], r60 * 1e3, color=col, lw=1, label=k)
    a3.set_ylabel("post-fit residual, 60-s means [mm/s]")
    a3.set_xlabel("hours into the pass")
    a3.set_title("coherent link: what the ground station and models limit")
    a3.legend(fontsize=7)
    a3.grid(alpha=0.3)
    fig.suptitle(f"S-band Doppler velocimetry, {site.get('name', 'site')} - simulation", fontsize=11)
    fig.savefig(os.path.join(out_dir, "doppler_sim.png"), dpi=110)
    with open(os.path.join(out_dir, "doppler_sim.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
