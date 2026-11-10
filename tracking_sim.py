#!/usr/bin/env python3
"""
End-to-end simulation: can this array track an aircraft on approach?

Physics simulated (sample by sample, 48 kHz):
  - aircraft flying a straight-in ILS approach (glide slope, constant ground speed),
  - broadband engine/airframe noise (pink, 50-2000 Hz) radiated omni-directionally,
  - propagation with the exact retarded time per sample (so Doppler and the motion during
    each frame are included), spherical spreading, optional uniform wind,
  - ground reflection: apex mic sees the grass (reflection coefficient R_GRASS) with the
    delay 2*h*sin(el)/c; ground mics see their board 16 mm below the axis,
  - uncorrelated ambient noise at every mic.
Processing (what the real tracker would do):
  - GCC-PHAT on every mic pair per 0.5 s frame, sub-sample peak interpolation,
  - the apex mic's ground echo arrives only 2*h*sin(el)/c after the direct sound (0.6 ms at
    5 deg elevation), inside one correlation-peak width, so it pulls the apex TDOAs and biases
    elevation by ~0.2 deg. Picking the "earliest" peak does not help (the peaks overlap).
    The fix used here: divide the predicted echo transfer function 1 + R exp(-j w delta) out of
    the cross-spectra (delta from the current elevation, R fitted per frame), iterate twice.
    Plain max-peak GCC is kept for comparison,
  - robust least-squares fit of a free 3D slowness vector -> direction of arrival,
  - range from the glide-path assumption (point on the ray closest to the glide path line).
Truth = direction to the aircraft at the EMISSION time (what ADS-B gives you once you
correct for the ~3-15 s sound travel time). The error of skipping that correction is plotted.

Run:  .venv/bin/python tracking_sim.py [--wind 5,0] [--apex 1.5] [--square 2.0] ...
Writes out/analysis/tracking_<tag>.png and prints error statistics by range.
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import array_params as AP

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "analysis")
P0 = 20e-6


def speed_of_sound(T_c):
    return 331.3 * np.sqrt(1 + T_c / 273.15)


class Approach:
    """Straight-in approach. Array frame: x east, y north, z up, metres."""

    def __init__(self, threshold_xy, heading_deg, gs_deg, tch, speed, start_dist, end_dist,
                 dev_lat=0.0, dev_vert=0.0):
        hd = np.radians(heading_deg)
        self.h = np.array([np.sin(hd), np.cos(hd), 0.0])  # direction of travel
        self.right = np.array([self.h[1], -self.h[0], 0.0])
        self.thr = np.array([threshold_xy[0], threshold_xy[1], 0.0])
        self.tg = np.tan(np.radians(gs_deg))
        self.tch, self.v, self.d0 = tch, speed, start_dist
        self.t_end = (start_dist - end_dist) / speed
        # the real aircraft flies offset from the ideal glide path (the range estimate doesn't know)
        self.dev_lat, self.dev_vert = dev_lat, dev_vert

    def pos(self, t):
        t = np.asarray(t, float)
        dist = self.d0 - self.v * t  # distance before threshold
        p = self.thr[None, :] - dist.reshape(-1)[:, None] * self.h[None, :] + self.dev_lat * self.right[None, :]
        p = p.reshape(t.shape + (3,)).copy()
        p[..., 2] = self.tch + dist * self.tg + self.dev_vert
        return p

    def glide_line(self):
        """Point and unit direction of the glide-path line."""
        d = self.h * 1.0
        d = d.copy()
        d[2] = -self.tg  # moving along h, altitude drops by tg per metre
        return self.thr + np.array([0, 0, self.tch]), d / np.linalg.norm(d)


def retarded_tau(ac, t, m, c, wind, iters=14):
    """Propagation delay tau(t) for receive times t at mic m (moving source, uniform wind)."""
    tau = np.linalg.norm(ac.pos(t) - m, axis=-1) / c
    for _ in range(iters):
        tau = np.linalg.norm(ac.pos(t - tau) + wind * tau[..., None] - m, axis=-1) / c
    return tau


def band_noise(n, fs, f_lo, f_hi, rng, pink=True):
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    g = ((f >= f_lo) & (f <= f_hi)).astype(float)
    if pink:
        g[1:] /= np.sqrt(f[1:] / f_lo)
    y = np.fft.irfft(X * g, n)
    return y / np.std(y)


def spectra(X, nfft):
    w = np.hanning(X.shape[1])
    return np.fft.rfft(X * w, nfft, axis=1)


def gcc_peak(G, f, fs, maxlag, f_lo, f_hi, interp=8, comp=None):
    """GCC-PHAT from a cross-spectrum G = X_i conj(X_j). Returns (tau = t_i - t_j, peak height).
    comp: optional complex echo transfer H_i conj(H_j) to divide out before PHAT weighting."""
    if comp is not None:
        G = G / comp
    nfft = 2 * (len(f) - 1)
    R = np.where((f >= f_lo) & (f <= f_hi), G / (np.abs(G) + 1e-20), 0)
    cc = np.fft.irfft(R, nfft * interp)
    fsi = fs * interp
    L = int(np.ceil(maxlag * fsi))
    cc = np.concatenate([cc[-L:], cc[:L + 1]])  # lags -L..L
    lags = np.arange(-L, L + 1)
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        a, b, c3 = cc[k - 1], cc[k], cc[k + 1]
        dk = 0.5 * (a - c3) / (a - 2 * b + c3)
    else:
        dk = 0.0
    return (lags[k] + dk) / fsi, cc[k]


def fit_slowness(mics, pairs, tdoa, weights=None, iters=5):
    """IRLS Huber fit of tdoa_ij = -(p_i - p_j).s  -> unit DOA vector and |s|*c."""
    A = -np.array([mics[i] - mics[j] for i, j in pairs])
    w = np.ones(len(pairs)) if weights is None else weights.copy()
    s = None
    for _ in range(iters):
        W = np.sqrt(w)[:, None]
        s, *_ = np.linalg.lstsq(A * W, tdoa * W[:, 0], rcond=None)
        r = tdoa - A @ s
        k = 1.5 * max(np.median(np.abs(r)) / 0.6745, 5e-6)
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return s


def closest_on_ray_to_line(o, u, q, d):
    """Point on ray o + r u (r>0) closest to line q + s d; returns r and the sine of the ray/line angle."""
    w0 = o - q
    a, b, c = u @ u, u @ d, d @ d
    dd, e = u @ w0, d @ w0
    den = a * c - b * b
    r = (b * e - c * dd) / den if den > 1e-12 else np.nan
    return r, np.sqrt(max(den, 0.0))


def run(args):
    rng = np.random.default_rng(args.seed)
    p = replace(AP.DEFAULT, square_side=args.square * 1000, apex_height=args.apex * 1000, mic6=args.mic6)
    g = AP.derive(p)
    mics = AP.mic_array_m(p, include_optional=args.mic6)
    kinds = [m["kind"] for m in AP.mic_table(p, args.mic6)]
    apex_i = kinds.index("apex")
    # reflecting plane height and reflection coefficient per mic
    z_ref = np.array([0.0 if k == "apex" else g.board_top / 1000 for k in kinds])
    R_ref = np.array([args.r_grass if k == "apex" else 1.0 for k in kinds])

    c_true = speed_of_sound(args.temp)
    c_proc = 343.0  # the tracker doesn't need c: the slowness fit is free
    wind = np.array([args.wind[0], args.wind[1], 0.0])
    ac = Approach(args.threshold, args.heading, args.gs, args.tch, args.speed, args.start_km * 1000,
                  args.end_km * 1000, args.dev_lat, args.dev_vert)

    fs, fs_src = args.fs, args.fs * 4
    N = int(args.frame * fs)
    nfft = 2 * N
    fgrid = np.fft.rfftfreq(nfft, 1 / fs)
    h_ref = mics[:, 2] - z_ref  # height of each capsule above its reflector
    R_grid = np.linspace(-0.9, 0.9, 19)
    pairs = [(i, j) for i in range(len(mics)) for j in range(i + 1, len(mics))]
    apex_pairs = np.array([apex_i in pr for pr in pairs])
    maxlag = [np.linalg.norm(mics[i] - mics[j]) / 300.0 + 2e-4 for i, j in pairs]
    src_amp = P0 * 10 ** (args.spl1m / 20)
    amb_amp = P0 * 10 ** (args.ambient / 20)

    frames = np.arange(args.frame * 0 + 15.0, ac.t_end, args.frame)  # receive times (start after sound arrives)
    res = {k: [] for k in ("t", "truth", "now", "est_max", "est_comp", "range", "snr", "gp_range", "gp_sin",
                           "R_est")}
    q_gp, d_gp = ac.glide_line()
    for t0 in frames:
        t = t0 + np.arange(N) / fs
        taus = []
        for mi, m in enumerate(mics):
            img = m.copy()
            img[2] = 2 * z_ref[mi] - m[2]
            taus.append((retarded_tau(ac, t, m, c_true, wind), retarded_tau(ac, t, img, c_true, wind)))
        te_all = np.concatenate([t - a for pr in taus for a in pr])
        lo, hi = te_all.min() - 0.01, te_all.max() + 0.01
        ns = int((hi - lo) * fs_src) + 1
        s = band_noise(ns, fs_src, 50, 2000, rng) * src_amp
        tgrid = lo + np.arange(ns) / fs_src
        X = np.empty((len(mics), N))
        for mi, m in enumerate(mics):
            sig = np.zeros(N)
            img = m.copy()
            img[2] = 2 * z_ref[mi] - m[2]
            for tau, pt, R in ((taus[mi][0], m, 1.0), (taus[mi][1], img, R_ref[mi])):
                dist = tau * c_true  # good enough for 1/r
                sig += R * np.interp(t - tau, tgrid, s) / dist
            X[mi] = sig + band_noise(N, fs, 50, 2000, rng) * amb_amp
        # --- processing ---
        Sx = spectra(X, nfft)
        cross = {pr: Sx[pr[0]] * np.conj(Sx[pr[1]]) for pr in pairs}

        def solve(Hs=None, interp=8):
            td, pk = [], []
            for pr, ml in zip(pairs, maxlag):
                comp = None if Hs is None else Hs[pr[0]] * np.conj(Hs[pr[1]])
                tt, hh = gcc_peak(cross[pr], fgrid, fs, ml, 80, 2000, interp=interp, comp=comp)
                td.append(tt)
                pk.append(hh)
            return fit_slowness(mics, pairs, np.array(td)), np.array(pk)

        u_max, _ = solve()  # plain GCC-PHAT, ignores the echoes
        # Echo compensation: each mic hears direct + R_m * echo delayed by 2 h_m sin(el) / c, where
        # h_m = height above its reflector (known) and el comes from the previous estimate. Ground
        # mics: R = 1 (hard board). Apex: R unknown (grass, possibly negative at grazing incidence),
        # chosen per frame as the value that makes the apex-pair correlation peaks sharpest.
        u = u_max / np.linalg.norm(u_max)
        for _ in range(2):
            delta = 2 * h_ref * max(u[2], 0.0) / c_proc

            def Hset(Ra):
                return [1 + (Ra if m == apex_i else 1.0) * np.exp(-2j * np.pi * fgrid * delta[m])
                        for m in range(len(mics))]

            scores = [solve(Hset(Ra), interp=1)[1][apex_pairs].sum() for Ra in R_grid]
            Ra_best = R_grid[int(np.argmax(scores))]
            s_c, _ = solve(Hset(Ra_best))
            u = s_c / np.linalg.norm(s_c)
        u_comp = u
        res["R_est"].append(Ra_best)
        tc = t0 + args.frame / 2
        tau0 = retarded_tau(ac, np.array([tc]), np.zeros(3), c_true, np.zeros(3))[0]
        pe = ac.pos(np.array([tc - tau0]))[0]
        pn = ac.pos(np.array([tc]))[0]
        res["t"].append(tc)
        res["truth"].append(pe / np.linalg.norm(pe))
        res["now"].append(pn / np.linalg.norm(pn))
        res["est_max"].append(u_max / np.linalg.norm(u_max))
        ue = u_comp
        res["est_comp"].append(ue)
        res["range"].append(np.linalg.norm(pe))
        r, sn = closest_on_ray_to_line(np.zeros(3), ue, q_gp, d_gp)
        res["gp_range"].append(r)
        res["gp_sin"].append(sn)
        sig_p = src_amp / np.linalg.norm(pe)
        res["snr"].append(20 * np.log10(sig_p / amb_amp))
    for k in res:
        res[k] = np.array(res[k])
    return res, p, ac


def azel(u):
    return np.degrees(np.arctan2(u[..., 0], u[..., 1])) % 360, np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))


def ang(u, v):
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1)))


def report(res, p, ac, args, tag):
    os.makedirs(OUT, exist_ok=True)
    e_max = ang(res["est_max"], res["truth"])
    e_comp = ang(res["est_comp"], res["truth"])
    e_now = ang(res["now"], res["truth"])
    rng_km = res["range"] / 1000
    az_t, el_t = azel(res["truth"])
    az_e, el_e = azel(res["est_comp"])
    az_m, el_m = azel(res["est_max"])

    print(f"\n=== {tag}: square {p.square_side / 1000:.1f} m, apex {p.apex_height / 1000:.1f} m, "
          f"wind {args.wind} m/s, {len(res['t'])} frames ===")
    print(f"{'range km':>10} {'n':>4} {'el deg':>8} {'SNR dB':>7} | {'max-peak med/p95':>17} | "
          f"{'echo-comp med/p95':>17} | {'no retard. corr':>15}")
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 20)]
    for lo, hi in bins:
        mk = (rng_km >= lo) & (rng_km < hi)
        if mk.sum() == 0:
            continue
        print(f"{lo:4.0f}-{hi:<4.0f}  {mk.sum():4d} {np.median(el_t[mk]):8.1f} {np.median(res['snr'][mk]):7.1f} | "
              f"{np.median(e_max[mk]):7.2f} / {np.percentile(e_max[mk], 95):6.2f}° | "
              f"{np.median(e_comp[mk]):7.2f} / {np.percentile(e_comp[mk], 95):6.2f}° | "
              f"{np.median(e_now[mk]):12.2f}°")
    gp_err = np.abs(res["gp_range"] - res["range"])
    good = res["gp_sin"] > 0.2
    if good.any():
        print(f"glide-path range estimate (ray/glide-path angle > 11.5 deg, {good.sum()} frames): "
              f"median error {np.median(gp_err[good]):.0f} m "
              f"({np.median(gp_err[good] / res['range'][good]) * 100:.1f} % of range)")
    if (~good).any():
        print(f"  ...poorly conditioned in {(~good).sum()} frames (ray nearly parallel to the glide path), "
              f"median error {np.nanmedian(gp_err[~good]):.0f} m")

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    t = res["t"]
    ax = axs[0, 0]
    ax.plot(t, az_t, "k-", lw=2, label="truth (emission time)")
    ax.plot(t, azel(res["now"])[0], "k:", label="ADS-B position at reception time")
    ax.plot(t, az_m, ".", ms=3, color="C3", label="estimate, max-peak")
    ax.plot(t, az_e, ".", ms=3, color="C0", label="estimate, echo-compensated")
    ax.set_ylabel("azimuth [deg]")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.plot(t, el_t, "k-", lw=2)
    ax.plot(t, azel(res["now"])[1], "k:")
    ax.plot(t, el_m, ".", ms=3, color="C3")
    ax.plot(t, el_e, ".", ms=3, color="C0")
    ax.set_ylabel("elevation [deg]")
    ax.set_xlabel("time [s]")
    ax = axs[0, 1]
    ax.semilogy(rng_km, e_max, ".", ms=3, color="C3", label="max-peak")
    ax.semilogy(rng_km, e_comp, ".", ms=3, color="C0", label="echo-compensated")
    ax.semilogy(rng_km, e_now, "k:", label="error if ADS-B not retarded")
    ax.set_xlabel("slant range [km]")
    ax.set_ylabel("angular error [deg]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.plot(rng_km, res["gp_range"] / 1000, ".", ms=3, color="C0", label="glide-path range estimate")
    ax.plot(rng_km, rng_km, "k-", label="truth")
    bad = res["gp_sin"] <= 0.2
    ax.plot(rng_km[bad], res["gp_range"][bad] / 1000, "x", ms=4, color="C3", label="ill-conditioned (ray ∥ glide path)")
    ax.set_xlabel("true slant range [km]")
    ax.set_ylabel("estimated range [km]")
    ax.set_ylim(0, rng_km.max() * 1.3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Approach tracking simulation — {tag}: square {p.square_side / 1000:.1f} m, apex "
                 f"{p.apex_height / 1000:.1f} m, wind {tuple(args.wind)} m/s, source {args.spl1m:.0f} dB@1 m, "
                 f"ambient {args.ambient:.0f} dB")
    fn = os.path.join(OUT, f"tracking_{tag}.png")
    fig.savefig(fn, dpi=100)
    plt.close(fig)
    print("plot:", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", type=float, default=2.0, help="m")
    ap.add_argument("--apex", type=float, default=1.5, help="m")
    ap.add_argument("--mic6", action="store_true")
    ap.add_argument("--threshold", type=lambda s: tuple(map(float, s.split(","))), default=(2500.0, -400.0),
                    help="runway threshold x,y in array frame [m]")
    ap.add_argument("--heading", type=float, default=90.0, help="landing direction, deg true")
    ap.add_argument("--gs", type=float, default=3.0)
    ap.add_argument("--tch", type=float, default=15.0, help="threshold crossing height [m]")
    ap.add_argument("--speed", type=float, default=70.0, help="ground speed [m/s]")
    ap.add_argument("--start-km", type=float, default=10.0, help="start distance before threshold")
    ap.add_argument("--end-km", type=float, default=1.0)
    ap.add_argument("--dev-lat", type=float, default=20.0, help="aircraft offset right of centreline [m]")
    ap.add_argument("--dev-vert", type=float, default=10.0, help="aircraft offset above glide path [m]")
    ap.add_argument("--spl1m", type=float, default=135.0, help="source level, dB SPL at 1 m (band 50-2000 Hz)")
    ap.add_argument("--ambient", type=float, default=45.0, help="ambient noise dB SPL per mic, same band")
    ap.add_argument("--r-grass", type=float, default=0.7, help="ground reflection coefficient under the apex mic")
    ap.add_argument("--wind", type=lambda s: tuple(map(float, s.split(","))), default=(0.0, 0.0), help="wx,wy m/s")
    ap.add_argument("--temp", type=float, default=15.0, help="air temperature C")
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--frame", type=float, default=0.5, help="s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="nominal")
    args = ap.parse_args()
    res, p, ac = run(args)
    report(res, p, ac, args, args.tag)


if __name__ == "__main__":
    main()
