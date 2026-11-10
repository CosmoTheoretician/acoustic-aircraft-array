#!/usr/bin/env python3
"""
Direction-of-arrival (DOA) accuracy of the array geometry, from timing error alone.

Model: far-field plane wave. Arrival time at mic i:  t_i = t0 - p_i . u / c,
u = unit vector array -> source (az from north, clockwise; el above horizon).
Unknowns (t0, az, el); timing noise is i.i.d. per mic with sigma_mic = sigma_tdoa / sqrt(2),
so that the pairwise TDOA error is sigma_tdoa (default 20 us). The Cramer-Rao bound
cov = sigma^2 (H^T H)^-1 then gives the 1-sigma angular error at every point of the sky.
At 1-5 km the plane-wave model is valid (checked below); the angular error therefore does
not depend on range, and the cross-track position error is simply angle x range.

Also reported (all for the default geometry):
  - bias from the plane-wave assumption at 1 km,
  - bias from mic position errors (the "1 cm repeatability" requirement),
  - bias from wind (wavefront drifts with the air: angle error ~ w_perp / c).

Run:  .venv/bin/python doa_analysis.py [--sigma-us 20] [--mic6]
Writes out/analysis/doa_skymaps.png, doa_vs_elevation.png, doa_summary.csv
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import array_params as AP

C = 343.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "analysis")


def unit_vectors(az, el):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    u = np.stack([ce * sa, ce * ca, se], -1)
    du_az = np.stack([ce * ca, -ce * sa, np.zeros_like(az)], -1)
    du_el = np.stack([-se * sa, -se * ca, ce], -1)
    return u, du_az, du_el


def crlb(mics, az, el, sigma_mic, c=C):
    """1-sigma errors (rad): great-circle angle, az, el. Known speed of sound."""
    u, daz, delv = unit_vectors(az, el)
    n = len(mics)
    H = np.empty(az.shape + (n, 3))
    H[..., 0] = 1.0
    H[..., 1] = -np.einsum("ij,...j->...i", mics, daz) / c
    H[..., 2] = -np.einsum("ij,...j->...i", mics, delv) / c
    F = np.einsum("...ia,...ib->...ab", H, H) / sigma_mic ** 2
    cov = np.linalg.inv(F)
    s_az = np.sqrt(cov[..., 1, 1])
    s_el = np.sqrt(cov[..., 2, 2])
    ang = np.sqrt(np.cos(el) ** 2 * cov[..., 1, 1] + cov[..., 2, 2])
    return ang, s_az, s_el


def crlb_free_c(mics, az, el, sigma_mic, c=C):
    """Same, but slowness vector fitted freely (speed of sound unknown -> immune to temperature/wind speed)."""
    u, _, _ = unit_vectors(az, el)
    H = np.hstack([np.ones((len(mics), 1)), -mics])
    cov_s = np.linalg.inv(H.T @ H)[1:, 1:] * sigma_mic ** 2
    Pm = np.eye(3) - u[..., :, None] * u[..., None, :]
    cu = c ** 2 * np.einsum("...ab,bc,...cd->...ad", Pm, cov_s, Pm)
    return np.sqrt(np.trace(cu, axis1=-2, axis2=-1))


def fit_plane_wave(mics, tau, c=C):
    """Unconstrained least-squares slowness fit from arrival times; returns unit vector."""
    H = np.hstack([np.ones((len(mics), 1)), -mics])
    sol, *_ = np.linalg.lstsq(H, tau, rcond=None)
    s = sol[1:]
    return s / np.linalg.norm(s)


def angle_between(u, v):
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-us", type=float, default=20.0, help="pairwise TDOA error (1 sigma), microseconds")
    ap.add_argument("--mic6", action="store_true", help="include the optional 6th mic")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sig = args.sigma_us * 1e-6 / np.sqrt(2)

    az = np.radians(np.arange(0, 361, 2.0))
    el = np.radians(np.arange(1, 90.1, 1.0))
    AZ, EL = np.meshgrid(az, el)
    band = (EL >= np.radians(2)) & (EL <= np.radians(30))  # where approaching aircraft are

    sides = [1500.0, 2000.0, 3000.0]
    heights = [1000.0, 1500.0, 2500.0]
    rows = []
    fig, axs = plt.subplots(len(heights), len(sides), subplot_kw=dict(projection="polar"),
                            figsize=(13, 13), constrained_layout=True)
    norm = LogNorm(vmin=0.08, vmax=0.5)
    for i, H in enumerate(heights):
        for j, L in enumerate(sides):
            p = replace(AP.DEFAULT, square_side=L, apex_height=H, mic6=args.mic6)
            mics = AP.mic_array_m(p, include_optional=args.mic6)
            ang, s_az, s_el = crlb(mics, AZ, EL, sig)
            angd = np.degrees(ang)
            free = np.degrees(crlb_free_c(mics, AZ, EL, sig))
            ax = axs[i, j]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            pc = ax.pcolormesh(AZ, 90 - np.degrees(EL), angd, norm=norm, cmap="viridis", shading="auto")
            ax.set_ylim(0, 89)
            ax.set_yticks([30, 60, 80])
            ax.set_yticklabels(["60°", "30°", "10°"], fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_title(f"square {L / 1000:.1f} m, apex {H / 1000:.1f} m\n"
                         f"median {np.median(angd[band]):.2f}°  (el 2-30°)", fontsize=9)
            rows.append(dict(square_m=L / 1000, apex_m=H / 1000,
                             band_median_deg=np.median(angd[band]), band_p95_deg=np.percentile(angd[band], 95),
                             band_az_median_deg=np.median(np.degrees(s_az * np.cos(EL))[band]),
                             band_el_median_deg=np.median(np.degrees(s_el)[band]),
                             sky_median_deg=np.median(angd), free_c_band_median_deg=np.median(free[band]),
                             xtrack_3km_m=np.median(ang[band]) * 3000, xtrack_5km_m=np.median(ang[band]) * 5000))
    fig.colorbar(pc, ax=axs, shrink=0.6, label=f"1σ angular error [deg] for {args.sigma_us:.0f} µs TDOA error")
    fig.suptitle("DOA error sky maps (zenith at centre, horizon at edge, north up)", fontsize=12)
    fig.savefig(os.path.join(OUT, "doa_skymaps.png"), dpi=110)
    plt.close(fig)

    # error vs elevation (median over azimuth)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for H in heights:
        for L in sides:
            p = replace(AP.DEFAULT, square_side=L, apex_height=H, mic6=args.mic6)
            mics = AP.mic_array_m(p, include_optional=args.mic6)
            ang, s_az, s_el = crlb(mics, AZ, EL, sig)
            ls = {1500.0: ":", 2000.0: "-", 3000.0: "--"}[L]
            col = {1000.0: "C0", 1500.0: "C1", 2500.0: "C2"}[H]
            ax[0].plot(np.degrees(el), np.degrees(np.median(s_el, 1)), ls, color=col,
                       label=f"{L / 1000:.1f} m / {H / 1000:.1f} m")
            ax[1].plot(np.degrees(el), np.degrees(np.median(s_az * np.cos(EL), 1)), ls, color=col)
    for a, t in zip(ax, ["elevation error (1σ)",
                         "azimuth error × cos(el) (1σ) — depends on square size only"]):
        a.set_xlabel("source elevation [deg]")
        a.set_ylabel("deg")
        a.set_yscale("log")
        a.set_title(t)
        a.axvspan(2, 30, color="0.9", zorder=0)
        a.grid(True, which="both", alpha=0.3)
    ax[0].legend(title="square / apex", fontsize=8)
    fig.savefig(os.path.join(OUT, "doa_vs_elevation.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT, "doa_summary.csv"), "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4g}" for v in r.values()) + "\n")
    print(f"TDOA error {args.sigma_us:.0f} us (1 sigma). Approach band = elevations 2-30 deg.")
    print(f"{'square':>6} {'apex':>5} | {'median':>7} {'p95':>6} {'az*cos':>7} {'el':>6} | {'freeC':>6} | "
          f"{'x-trk 3km':>9} {'5km':>6}")
    for r in rows:
        print(f"{r['square_m']:6.1f} {r['apex_m']:5.1f} | {r['band_median_deg']:7.3f} {r['band_p95_deg']:6.3f} "
              f"{r['band_az_median_deg']:7.3f} {r['band_el_median_deg']:6.3f} | {r['free_c_band_median_deg']:6.3f} | "
              f"{r['xtrack_3km_m']:8.1f}m {r['xtrack_5km_m']:5.1f}m")

    # ---- systematic error checks on the default geometry ----
    p = replace(AP.DEFAULT, mic6=args.mic6)
    mics = AP.mic_array_m(p, include_optional=args.mic6)
    azc, elc = np.meshgrid(np.radians(np.arange(0, 360, 10.0)), np.radians(np.arange(2, 60, 2.0)))
    u, _, _ = unit_vectors(azc.ravel(), elc.ravel())
    # (1) plane-wave assumption at 1 km
    biases = []
    for uu in u:
        src = uu * 1000.0
        tau = np.linalg.norm(mics - src, axis=1) / C
        biases.append(angle_between(fit_plane_wave(mics, tau), uu))
    print(f"\nplane-wave model bias at 1 km: max {max(biases):.4f} deg  (negligible -> far-field OK)")
    # (2) mic position errors
    rng = np.random.default_rng(1)
    errs = []
    for sig_pos in (0.002, 0.005, 0.01):
        e = []
        for _ in range(200):
            dp = rng.normal(0, sig_pos, mics.shape)
            for uu in u[::7]:
                tau = -(mics + dp) @ uu / C  # truth with displaced mics, processed with nominal mics
                e.append(angle_between(fit_plane_wave(mics, tau), uu))
        errs.append((sig_pos, np.median(e), np.percentile(e, 95)))
        print(f"mic position error {sig_pos * 1000:.0f} mm/axis (1σ): DOA bias median {np.median(e):.3f} deg, "
              f"p95 {np.percentile(e, 95):.3f} deg   (= {sig_pos / C * 1e6:.0f} us per axis)")
    # (3) wind
    for w in (2.0, 5.0, 10.0):
        print(f"cross-wind {w:.0f} m/s: apparent-direction bias up to {np.degrees(w / C):.2f} deg at any range "
              f"(wavefront centre drifts by w*R/c); a free-c fit does not remove it")


if __name__ == "__main__":
    main()
