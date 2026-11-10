#!/usr/bin/env python3
"""
End-to-end simulation: can this array track aircraft on the ILS approach to Luxembourg
(ELLX) runway 24 from a given garden site?

Physics simulated (sample by sample, 48 kHz):
  - aircraft on the RWY 24 ILS (3.0 deg glide path, straight in), optionally offset from
    the ideal path, flying at constant ground speed,
  - broadband engine/airframe noise (pink, 50-3000 Hz) radiated omni-directionally,
  - exact retarded time per sample (Doppler and motion within each frame included),
    spherical spreading, atmospheric absorption (ISO 9613-1, ~10 C / 70 % RH), uniform wind,
  - ground reflection for every mic (the array is elevated; reflection coefficient R_GRASS),
    delay 2*h*sin(el)/c with h = capsule height above the ground,
  - uncorrelated ambient noise at every mic.
Processing (what the real tracker would do):
  - GCC-PHAT on every mic pair per 0.5 s frame, sub-sample peak interpolation,
  - ground-echo compensation: the echo transfer 1 + R exp(-j w delta_m) of every mic is
    divided out of the cross-spectra, delta_m = 2 h_m s_z from the fitted slowness (no speed
    of sound needed), iterated twice. R (the lawn's reflection coefficient) is learned from the
    most coherent frames and then held fixed; per-frame R fits are unreliable at low SNR. Base-base pairs have identical echoes, which cancel anyway;
    the apex pairs are the ones that need it. delta = 2 h s_z from the fitted slowness. Plain max-peak GCC is kept for comparison,
  - robust least-squares fit of a free 3D slowness vector -> wavefront direction,
  - wind correction with the anemometer: direction to the source = normalise(u - w/c).
    The anemometer sits near the ground and sees only a fraction of the wind averaged along
    the sound path (wind grows with height), so this removes most but not all of the bias,
  - range from the glide-path assumption (point on the ray closest to the glide path line).
Truth = direction to the aircraft at the EMISSION time (ADS-B position corrected for the
3-30 s sound travel time). The error of skipping that correction is plotted too.

Site: by default a generic EXAMPLE site 6 km out, 1.5 km south-east of the centreline.
Your real site goes in a git-ignored JSON file:  {"name": "...", "lat": .., "lon": .., "ground_elev_m": ..}
    .venv/bin/python tracking_sim.py --site site_local.json [--wind-speed 5 --wind-from 240]
Results for a local site are written to out_local/ (also git-ignored).
"""

import argparse
import json
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import array_params as AP

HERE = os.path.dirname(os.path.abspath(__file__))
P0 = 20e-6

# ELLX runway data (OurAirports; true course from the threshold coordinates, matches the
# published 240.2 deg true). Elevations in metres AMSL. Check against the current AIP.
ELLX = {
    "24": dict(lat=49.635502, lon=6.23556, elev=1214 * 0.3048, course=240.22, gs=3.0, tch=15.0),
    "06": dict(lat=49.617699, lon=6.18746, elev=1158 * 0.3048, course=60.22, gs=3.0, tch=15.0),
}

# Atmospheric absorption, dB/km, ISO 9613-1 at ~10 C, 70 % RH
ABS_F = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000])
ABS_DB = np.array([0.03, 0.12, 0.41, 1.04, 1.93, 3.66, 9.66, 32.8, 117.0])

# ---------------------------------------------------------------------------- geodesy
A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)


def lla2ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    N = A_WGS / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo),
                     (N * (1 - E2) + h) * np.sin(la)])


def enu_basis(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    e = np.array([-np.sin(lo), np.cos(lo), 0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return np.vstack([e, n, u])  # rows: e, n, u  (ECEF -> ENU)


def example_site():
    """Generic public example: 6 km out from THR 24 along the approach, 1.5 km to the
    south-east (left of inbound aircraft), ground 300 m AMSL."""
    r = ELLX["24"]
    out = np.radians(r["course"] + 180)  # outbound along the approach
    left = np.radians(r["course"] - 90)  # left of the inbound aircraft
    de = 6000 * np.sin(out) + 1500 * np.sin(left)
    dn = 6000 * np.cos(out) + 1500 * np.cos(left)
    lat = r["lat"] + dn / 111_200
    lon = r["lon"] + de / (111_320 * np.cos(np.radians(r["lat"])))
    return dict(name="example site (6 km out, 1.5 km SE of centreline)", lat=lat, lon=lon, ground_elev_m=300.0)


# ---------------------------------------------------------------------------- scene
def speed_of_sound(T_c):
    return 331.3 * np.sqrt(1 + T_c / 273.15)


class Approach:
    """Straight-in ILS approach, expressed in the array (site ENU) frame, metres."""

    def __init__(self, site, rwy, speed, start_dist, end_dist, dev_lat=0.0, dev_vert=0.0):
        r = ELLX[rwy]
        R_site = enu_basis(site["lat"], site["lon"])
        R_thr = enu_basis(r["lat"], r["lon"])
        o = lla2ecef(site["lat"], site["lon"], site["ground_elev_m"])
        self.thr = R_site @ (lla2ecef(r["lat"], r["lon"], r["elev"]) - o)
        crs = np.radians(r["course"])
        to_site = R_site @ R_thr.T  # threshold ENU -> site ENU
        self.h = to_site @ np.array([np.sin(crs), np.cos(crs), 0.0])  # direction of travel
        self.up = to_site @ np.array([0, 0, 1.0])
        self.right = np.cross(self.h, self.up)
        self.tg = np.tan(np.radians(r["gs"]))
        self.tch, self.v, self.d0 = r["tch"], speed, start_dist
        self.t_end = (start_dist - end_dist) / speed
        self.dev_lat, self.dev_vert = dev_lat, dev_vert

    def pos(self, t):
        t = np.asarray(t, float)
        dist = (self.d0 - self.v * t).reshape(-1, 1)  # distance before threshold
        p = (self.thr[None] - dist * self.h[None] + (self.tch + dist * self.tg + self.dev_vert) * self.up[None]
             + self.dev_lat * self.right[None])
        return p.reshape(t.shape + (3,))

    def glide_line(self):
        d = self.h - self.tg * self.up
        return self.thr + self.tch * self.up, d / np.linalg.norm(d)


def retarded_tau(ac, t, m, c, wind, iters=14):
    """Propagation delay tau(t) for receive times t at point m (moving source, uniform wind)."""
    tau = np.linalg.norm(ac.pos(t) - m, axis=-1) / c
    for _ in range(iters):
        tau = np.linalg.norm(ac.pos(t - tau) + wind * tau[..., None] - m, axis=-1) / c
    return tau


def band_noise(n, fs, f_lo, f_hi, rng, pink=True, gain_db=None):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    g = ((f >= f_lo) & (f <= f_hi)).astype(float)
    if pink:
        g[1:] /= np.sqrt(f[1:] / f_lo)
    y = np.fft.irfft(X * g, n)
    y /= np.std(y)  # unit rms at the source (1 m), before propagation losses
    if gain_db is not None:
        y = np.fft.irfft(np.fft.rfft(y) * 10 ** (gain_db(f) / 20), n)
    return y


def absorption_db(f, r_m):
    lf = np.log(np.clip(f, ABS_F[0], ABS_F[-1]))
    return -np.exp(np.interp(lf, np.log(ABS_F), np.log(ABS_DB))) * r_m / 1000


# ---------------------------------------------------------------------------- processing
def spectra(X, nfft):
    return np.fft.rfft(X * np.hanning(X.shape[1]), nfft, axis=1)


def gcc_peak(G, f, fs, maxlag, f_lo, f_hi, interp=8, comp=None):
    """GCC-PHAT from a cross-spectrum G = X_i conj(X_j). Returns (tau = t_i - t_j, peak height)."""
    if comp is not None:
        G = G / comp
    nfft = 2 * (len(f) - 1)
    R = np.where((f >= f_lo) & (f <= f_hi), G / (np.abs(G) + 1e-20), 0)
    cc = np.fft.irfft(R, nfft * interp)
    fsi = fs * interp
    L = int(np.ceil(maxlag * fsi))
    cc = np.concatenate([cc[-L:], cc[:L + 1]])
    lags = np.arange(-L, L + 1)
    k = int(np.argmax(cc))
    dk = 0.0
    if 0 < k < len(cc) - 1:
        a, b, c3 = cc[k - 1], cc[k], cc[k + 1]
        den = a - 2 * b + c3
        dk = 0.5 * (a - c3) / den if den != 0 else 0.0
    return (lags[k] + dk) / fsi, cc[k]


def fit_slowness(mics, pairs, tdoa, iters=5):
    """IRLS Huber fit of tdoa_ij = -(p_i - p_j).s  -> slowness vector s."""
    A = -np.array([mics[i] - mics[j] for i, j in pairs])
    w = np.ones(len(pairs))
    s = None
    for _ in range(iters):
        W = np.sqrt(w)
        s, *_ = np.linalg.lstsq(A * W[:, None], tdoa * W, rcond=None)
        r = tdoa - A @ s
        k = 1.5 * max(np.median(np.abs(r)) / 0.6745, 5e-6)
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return s


def closest_on_ray_to_line(o, u, q, d):
    """Distance along ray o + r u closest to line q + s d, and sine of the ray/line angle."""
    w0 = o - q
    a, b, c = u @ u, u @ d, d @ d
    dd, e = u @ w0, d @ w0
    den = a * c - b * b
    r = (b * e - c * dd) / den if den > 1e-12 else np.nan
    return r, np.sqrt(max(den, 0.0))


# ---------------------------------------------------------------------------- run
def run(args, site):
    rng = np.random.default_rng(args.seed)
    p = replace(AP.DEFAULT, square_side=args.square * 1000, apex_rise=args.apex_rise * 1000,
                base_height=args.base_height * 1000, mic6=args.mic6)
    mics = AP.mic_array_m(p, include_optional=args.mic6)
    kinds = [m["kind"] for m in AP.mic_table(p, args.mic6)]
    apex_i = kinds.index("apex")
    h_ref = mics[:, 2].copy()  # every capsule's height above its reflector (the ground)

    c_true = speed_of_sound(args.temp)
    c_proc = 343.0
    wf = np.radians(args.wind_from)
    wind = -args.wind_speed * np.array([np.sin(wf), np.cos(wf), 0.0])  # blowing towards
    w_meas = wind * args.wind_seen + rng.normal(0, 0.2, 3) * np.array([1, 1, 0])
    ac = Approach(site, args.runway, args.speed, args.start_km * 1000, args.end_km * 1000,
                  args.dev_lat, args.dev_vert)

    fs, fs_src = args.fs, args.fs * 4
    N = int(args.frame * fs)
    nfft = 2 * N
    fgrid = np.fft.rfftfreq(nfft, 1 / fs)
    R_grid = np.linspace(-0.9, 0.9, 19)
    pairs = [(i, j) for i in range(len(mics)) for j in range(i + 1, len(mics))]
    apex_pairs = np.array([apex_i in pr for pr in pairs])
    maxlag = [np.linalg.norm(mics[i] - mics[j]) / 300.0 + 2e-4 for i, j in pairs]
    src_amp = P0 * 10 ** (args.spl1m / 20)
    amb_amp = P0 * 10 ** (args.ambient / 20)

    t_first = retarded_tau(ac, np.array([0.0]), np.zeros(3), c_true, wind)[0]
    frames = np.arange(t_first + 1.0, ac.t_end, args.frame)
    keys = ("t", "truth", "now", "est_max", "est_comp", "est_wind", "range", "snr", "gp_range", "gp_sin",
            "R_est", "pos", "s_max", "coh")
    res = {k: [] for k in keys}
    q_gp, d_gp = ac.glide_line()
    band = (fgrid >= 60) & (fgrid <= 2500)
    fb = fgrid[band]
    store = []

    def Hset(Ra, delta):
        return [1 + Ra * np.exp(-2j * np.pi * fb * delta[m]) for m in range(len(mics))]

    def solve(cross, Hs=None, interp=8):
        td, pk = [], []
        for k, ((i, j), ml) in enumerate(zip(pairs, maxlag)):
            G = np.zeros(len(fgrid), complex)
            G[band] = cross[k] if Hs is None else cross[k] / (Hs[i] * np.conj(Hs[j]))
            tt, hh = gcc_peak(G, fgrid, fs, ml, 60, 2500, interp=interp)
            td.append(tt)
            pk.append(hh)
        return fit_slowness(mics, pairs, np.array(td)), np.array(pk)

    for t0 in frames:
        t = t0 + np.arange(N) / fs
        taus = []
        for m in mics:
            img = m * np.array([1, 1, -1])
            taus.append((retarded_tau(ac, t, m, c_true, wind), retarded_tau(ac, t, img, c_true, wind)))
        te_all = np.concatenate([t - a for pr in taus for a in pr])
        lo, hi = te_all.min() - 0.01, te_all.max() + 0.01
        ns = int((hi - lo) * fs_src) + 1
        r_frame = taus[0][0].mean() * c_true
        s = band_noise(ns, fs_src, 50, 3000, rng, gain_db=lambda f: absorption_db(f, r_frame)) * src_amp
        tgrid = lo + np.arange(ns) / fs_src
        X = np.empty((len(mics), N))
        for mi in range(len(mics)):
            sig = np.zeros(N)
            for tau, R in ((taus[mi][0], 1.0), (taus[mi][1], args.r_grass)):
                sig += R * np.interp(t - tau, tgrid, s) / (tau * c_true)
            X[mi] = sig + band_noise(N, fs, 50, 3000, rng) * amb_amp
        # ---- processing, pass 1: per-frame GCC; keep the band-limited cross-spectra ----
        Sx = spectra(X, nfft)
        cross = np.array([(Sx[i] * np.conj(Sx[j]))[band] for i, j in pairs])
        s_max, pk = solve(cross)
        s_c = s_max
        Ra_best = 0.0
        for _ in range(2):
            delta = 2 * h_ref * max(s_c[2], 0.0)
            scores = [solve(cross, Hset(Ra, delta), interp=1)[1][apex_pairs].sum() for Ra in R_grid]
            Ra_best = R_grid[int(np.argmax(scores))]
            s_c, _ = solve(cross, Hset(Ra_best, delta))
        store.append(cross)

        tc = t0 + args.frame / 2
        tau0 = retarded_tau(ac, np.array([tc]), np.zeros(3), c_true, wind)[0]
        pe = ac.pos(np.array([tc - tau0]))[0]
        pn = ac.pos(np.array([tc]))[0]
        res["t"].append(tc)
        res["pos"].append(pe)
        res["truth"].append(pe / np.linalg.norm(pe))
        res["now"].append(pn / np.linalg.norm(pn))
        res["est_max"].append(s_max / np.linalg.norm(s_max))
        res["s_max"].append(s_max)
        res["coh"].append(pk.mean())
        res["range"].append(np.linalg.norm(pe))
        res["R_est"].append(Ra_best)
        sig_rms = src_amp / np.linalg.norm(pe) * 10 ** (np.mean(absorption_db(np.array([250, 500, 1000]),
                                                                               np.linalg.norm(pe))) / 20)
        res["snr"].append(20 * np.log10(sig_rms / amb_amp))

    # ---- pass 2: ground reflection coefficient is a property of the lawn, not of the frame.
    # Learn it from the most coherent frames (in deployment: from earlier aircraft too),
    # then compensate every frame with that fixed value.
    coh = np.array(res["coh"])
    good = coh >= np.percentile(coh, 70)
    R_fixed = float(np.median(np.array(res["R_est"])[good]))
    for k, cross in enumerate(store):
        s_c = res["s_max"][k]
        for _ in range(2):
            s_c, _ = solve(cross, Hset(R_fixed, 2 * h_ref * max(s_c[2], 0.0)))
        u = s_c / np.linalg.norm(s_c)
        u0 = res["s_max"][k] / np.linalg.norm(res["s_max"][k])
        if np.degrees(np.arccos(np.clip(u @ u0, -1, 1))) > 0.5:
            u = u0  # the echo bias is a few tenths of a degree at most; a bigger jump is noise
        u_w = u - w_meas / c_proc
        u_w /= np.linalg.norm(u_w)
        res["est_comp"].append(u)
        res["est_wind"].append(u_w)
        r, sn = closest_on_ray_to_line(np.zeros(3), u_w, q_gp, d_gp)
        res["gp_range"].append(r)
        res["gp_sin"].append(sn)
    res["R_fixed"] = R_fixed
    for k in res:
        res[k] = np.array(res[k])
    return res, p, ac, wind, w_meas


def azel(u):
    return np.degrees(np.arctan2(u[..., 0], u[..., 1])) % 360, np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))


def ang(u, v):
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1)))


def report(res, p, ac, args, tag, site, outdir, wind, w_meas):
    os.makedirs(outdir, exist_ok=True)
    e_max = ang(res["est_max"], res["truth"])
    e_comp = ang(res["est_comp"], res["truth"])
    e_wind = ang(res["est_wind"], res["truth"])
    e_now = ang(res["now"], res["truth"])
    rng_km = res["range"] / 1000
    az_t, el_t = azel(res["truth"])
    k0 = int(np.argmin(res["range"]))
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"=== {tag}: {site['name']} — ELLX RWY {args.runway} ILS ===")
    out(f"array: square {p.square_side / 1000:.1f} m, apex rise {p.apex_rise / 1000:.1f} m, base at "
        f"{p.base_height / 1000:.1f} m; threshold {np.linalg.norm(ac.thr[:2]) / 1000:.2f} km away, "
        f"{ac.thr[2]:.0f} m above site ground")
    out(f"closest approach: {res['range'][k0] / 1000:.2f} km slant, elevation {el_t[k0]:.1f} deg, azimuth "
        f"{az_t[k0]:.0f} deg; elevations seen {el_t.min():.1f}-{el_t.max():.1f} deg")
    out(f"ground reflection: true R = {args.r_grass:+.2f}, learned R = {res['R_fixed']:+.2f}")
    out(f"wind {np.linalg.norm(wind):.1f} m/s from {args.wind_from:.0f} deg; anemometer sees "
        f"{args.wind_seen * 100:.0f} % of the path-mean wind (+ 0.2 m/s noise)")
    out(f"{'range km':>9} {'n':>4} {'el':>5} {'SNR':>5} | {'max-peak':>13} | {'+echo comp':>13} | "
        f"{'+wind corr':>13} | {'no retard.':>9}")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 30)]:
        mk = (rng_km >= lo) & (rng_km < hi)
        if mk.sum() == 0:
            continue
        out(f"{lo:3.0f}-{hi:<4.0f} {mk.sum():5d} {np.median(el_t[mk]):5.1f} {np.median(res['snr'][mk]):5.0f} | "
            f"{np.median(e_max[mk]):5.2f}/{np.percentile(e_max[mk], 95):5.2f}° | "
            f"{np.median(e_comp[mk]):5.2f}/{np.percentile(e_comp[mk], 95):5.2f}° | "
            f"{np.median(e_wind[mk]):5.2f}/{np.percentile(e_wind[mk], 95):5.2f}° | {np.median(e_now[mk]):8.2f}°")
    out("(median/p95 angular error vs. the aircraft position at emission time)")
    gp_err = np.abs(res["gp_range"] - res["range"])
    good = res["gp_sin"] > 0.2
    if good.any():
        out(f"glide-path range, well-conditioned frames ({good.sum()}): median error {np.median(gp_err[good]):.0f} m "
            f"({np.median(gp_err[good] / res['range'][good]) * 100:.1f} %)")
    if (~good).any():
        out(f"glide-path range, ill-conditioned frames ({(~good).sum()}): median error {np.nanmedian(gp_err[~good]):.0f} m")
    with open(os.path.join(outdir, f"tracking_{tag}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axs = plt.subplots(2, 3, figsize=(19, 9.5), constrained_layout=True)
    t = res["t"]
    az_m, el_m = azel(res["est_max"])
    az_w, el_w = azel(res["est_wind"])
    ax = axs[0, 0]
    ax.plot(t, az_t, "k-", lw=2, label="truth (emission time)")
    ax.plot(t, azel(res["now"])[0], "k:", label="ADS-B at reception time")
    ax.plot(t, az_m, ".", ms=3, color="C3", label="estimate, plain GCC")
    ax.plot(t, az_w, ".", ms=3, color="C0", label="estimate, echo + wind corrected")
    ax.set_ylabel("azimuth [deg]")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.plot(t, el_t, "k-", lw=2)
    ax.plot(t, azel(res["now"])[1], "k:")
    ax.plot(t, el_m, ".", ms=3, color="C3")
    ax.plot(t, el_w, ".", ms=3, color="C0")
    ax.set_ylabel("elevation [deg]")
    ax.set_xlabel("time [s]")
    ax = axs[0, 1]
    ax.semilogy(rng_km, e_max, ".", ms=3, color="C3", label="plain GCC")
    ax.semilogy(rng_km, e_comp, ".", ms=3, color="C1", label="+ echo compensation")
    ax.semilogy(rng_km, e_wind, ".", ms=3, color="C0", label="+ wind correction")
    ax.semilogy(rng_km, e_now, "k:", label="ADS-B not retarded")
    ax.set_xlabel("slant range [km]")
    ax.set_ylabel("angular error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.plot(rng_km, res["gp_range"] / 1000, ".", ms=3, color="C0", label="glide-path range estimate")
    ax.plot(rng_km, rng_km, "k-", label="truth")
    bad = res["gp_sin"] <= 0.2
    ax.plot(rng_km[bad], res["gp_range"][bad] / 1000, "x", ms=4, color="C3", label="ill-conditioned")
    ax.set_xlabel("true slant range [km]")
    ax.set_ylabel("estimated range [km]")
    ax.set_ylim(0, rng_km.max() * 1.3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[0, 2]
    P = res["pos"]
    ax.plot(P[:, 0] / 1000, P[:, 1] / 1000, "-", color="C0", label="aircraft track (emission)")
    ax.plot([0], [0], "r^", ms=10, label="array")
    ax.plot([ac.thr[0] / 1000], [ac.thr[1] / 1000], "ks", label=f"THR {args.runway}")
    rw = ac.thr + ac.h * 4000
    ax.plot([ac.thr[0] / 1000, rw[0] / 1000], [ac.thr[1] / 1000, rw[1] / 1000], "k-", lw=4, label="runway")
    ax.set_aspect("equal")
    ax.set_xlabel("east [km]")
    ax.set_ylabel("north [km]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1, 2]
    ax.plot(rng_km, res["snr"], ".", ms=3)
    ax.set_xlabel("slant range [km]")
    ax.set_ylabel("band SNR at a mic [dB] (250-1000 Hz)")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{tag}: {site['name']} — ELLX RWY {args.runway}; square {p.square_side / 1000:.1f} m, apex rise "
                 f"{p.apex_rise / 1000:.1f} m, base {p.base_height / 1000:.1f} m; wind {np.linalg.norm(wind):.0f} m/s "
                 f"from {args.wind_from:.0f}°; source {args.spl1m:.0f} dB@1 m, ambient {args.ambient:.0f} dB")
    fn = os.path.join(outdir, f"tracking_{tag}.png")
    fig.savefig(fn, dpi=95)
    plt.close(fig)
    print("plot:", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="JSON with name, lat, lon, ground_elev_m (default: public example site)")
    ap.add_argument("--runway", default="24", choices=list(ELLX))
    ap.add_argument("--square", type=float, default=2.0, help="m")
    ap.add_argument("--apex-rise", type=float, default=1.4, help="m")
    ap.add_argument("--base-height", type=float, default=2.0, help="m")
    ap.add_argument("--mic6", action="store_true")
    ap.add_argument("--speed", type=float, default=72.0, help="ground speed [m/s]")
    ap.add_argument("--start-km", type=float, default=18.0, help="start distance before threshold")
    ap.add_argument("--end-km", type=float, default=0.5)
    ap.add_argument("--dev-lat", type=float, default=20.0, help="aircraft offset right of centreline [m]")
    ap.add_argument("--dev-vert", type=float, default=10.0, help="aircraft offset above glide path [m]")
    ap.add_argument("--spl1m", type=float, default=140.0, help="source level, dB SPL at 1 m (50-3000 Hz)")
    ap.add_argument("--ambient", type=float, default=45.0, help="ambient noise dB SPL per mic, same band")
    ap.add_argument("--r-grass", type=float, default=0.7, help="ground reflection coefficient")
    ap.add_argument("--wind-speed", type=float, default=0.0, help="m/s")
    ap.add_argument("--wind-from", type=float, default=240.0, help="deg true (RWY 24 is used in westerlies)")
    ap.add_argument("--wind-seen", type=float, default=0.7,
                    help="fraction of the path-mean wind that the pole-top anemometer measures")
    ap.add_argument("--temp", type=float, default=10.0, help="air temperature C")
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--frame", type=float, default=0.5, help="s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="nominal")
    args = ap.parse_args()
    if args.site:
        with open(args.site) as f:
            site = json.load(f)
        outdir = os.path.join(HERE, "out_local", "analysis")
    else:
        site = example_site()
        outdir = os.path.join(HERE, "out", "analysis")
    res, p, ac, wind, w_meas = run(args, site)
    report(res, p, ac, args, args.tag, site, outdir, wind, w_meas)


if __name__ == "__main__":
    main()
