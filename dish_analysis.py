#!/usr/bin/env python3
"""
Dish extension, phase 1 analysis: link budget (S-band Doppler targets), radio-astronomy
sensitivity (L-band), wind loads, drive sizing, foundation sizing, power.

Everything comes from dish_params.py. The spacecraft numbers are the best public data found
(IM-1 FCC filings and recordings); they are labelled as such and must be replaced with
IM-3's own filing when it is published.

Run:  .venv/bin/python dish_analysis.py      -> out/dish/dish_analysis.txt
"""

import math
import os

import numpy as np

import dish_params as DP

K_B = 1.380649e-23
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "dish")
RHO = 1.25

# EIRP in dBW, carrier fraction in dB (0 = pure carrier). Sources / assumptions in `note`.
TARGETS = [
    dict(name="IM-3 low-gain antenna, lander upright", eirp=10 * math.log10(8) + 5, carrier_db=-5.3,
         f="s", note="IM-1 FCC: 8 W transceiver, 5 dBic LGA; residual carrier, mod index ~1 rad (assumed)"),
    dict(name="IM-3 low-gain antenna, lander tipped (-15 dB)", eirp=10 * math.log10(8) + 5 - 15,
         carrier_db=-5.3, f="s", note="IM-1 and IM-2 both ended on their side"),
    dict(name="IM-3 high-gain antenna", eirp=10 * math.log10(25) + 15, carrier_db=-5.3, f="s",
         note="IM-1 FCC: 25 W transmitter, 15 dBic HGA"),
    dict(name="LRO S-band, omni (ASSUMED 5 W, 0 dBi)", eirp=10 * math.log10(5), carrier_db=-3.0,
         f="lro", note="assumption; replace with measured C/N0 after the first LRO pass"),
]


def db(x):
    return 10 * math.log10(x)


def tsys(p, nf_db, t_ant):
    L = 10 ** (p.feed_loss_db / 10)
    t_lna = 290 * (10 ** (nf_db / 10) - 1)
    return t_ant / L + 290 * (1 - 1 / L) + t_lna


def moon_fill(g, f_mhz):
    om_beam = 1.133 * g.hpbw_deg(f_mhz) ** 2
    om_moon = math.pi * 0.26 ** 2
    return om_moon / om_beam


def q_peak(p, z_m):
    """EN 1991-1-4 peak velocity pressure (Pa) at height z_m, terrain roughness z0."""
    z0 = p.terrain_z0
    z = max(z_m, 5.0)
    kr = 0.19 * (z0 / 0.05) ** 0.07
    cr = kr * math.log(z / z0)
    iv = 1 / math.log(z / z0)
    return (1 + 7 * iv) * 0.5 * RHO * (cr * p.vb0) ** 2


def main():
    p = DP.DEFAULT
    g = DP.DishGeometry(p)
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=== Dish extension, phase 1 analysis ===")
    out(f"reflector {p.D / 1000:.1f} m, f/D {p.f_over_D}, f = {g.f:.0f} mm, depth {g.depth:.0f} mm; "
        f"dish + feed {g.m_dish:.0f} kg balanced by a {g.cw_mass:.0f} kg counterweight at {p.cw_arm:.0f} mm")

    # ------------------------------------------------------------------ S-band link budget
    out("\n--- S-band link budget (Doppler velocimetry) ---")
    r_max = 405_500e3  # lunar apogee
    for key, f_mhz in (("s", p.f_s_mhz), ("lro", p.f_lro_mhz)):
        lam = DP.C / (f_mhz * 1e6)
        gdb = g.gain_dbi(f_mhz)
        t_moon = 220.0 * moon_fill(g, f_mhz)
        ts = tsys(p, p.lna_nf_db_s, p.t_sky_k + p.t_spill_k + t_moon)
        gt = gdb - db(ts)
        fspl = 20 * math.log10(4 * math.pi * r_max / lam)
        if key == "s":
            out(f"{f_mhz:.1f} MHz: G = {gdb:.1f} dBi, HPBW {g.hpbw_deg(f_mhz):.2f} deg, Tsys = {ts:.0f} K "
                f"(Moon in beam {t_moon:.1f} K), G/T = {gt:.1f} dB/K, path loss at apogee {fspl:.1f} dB")
        losses = 0.2 + 0.1 + 0.3  # pointing + atmosphere + polarisation
        for t in TARGETS:
            if t["f"] != key:
                continue
            cn0 = t["eirp"] - fspl + gt - losses - 10 * math.log10(K_B)
            cn0_c = cn0 + t["carrier_db"]
            T = 1.0
            sig_f = math.sqrt(3 / (math.pi ** 2 * 10 ** (cn0_c / 10) * T ** 3))
            sig_v = sig_f * lam
            dec = "yes" if cn0 >= 37.4 else "no"
            dop = "yes" if cn0_c >= 13 else ("10 s avg" if cn0_c >= 3 else "no")
            out(f"  {t['name']:44s} C/N0 {cn0:5.1f} dB-Hz, carrier {cn0_c:5.1f} dB-Hz | "
                f"1-s Doppler noise {sig_v * 1e3:6.2f} mm/s | carrier Doppler: {dop:8s} | decode 1.26 kbps: {dec}")
            out(f"    ({t['note']})")
    out("  thresholds: carrier Doppler at 1-s points needs ~13 dB-Hz in the carrier; decoding the IM-1 "
        "style 2.5 kbaud r=1/2 link needs ~37 dB-Hz")

    # ------------------------------------------------------------------ L-band astronomy
    out("\n--- L-band radio astronomy (21 cm feed) ---")
    fl = p.f_hi_mhz
    gl = g.gain_dbi(fl)
    tl = tsys(p, p.lna_nf_db_l, 5.0 + p.t_spill_k)
    ae = 10 ** (gl / 10) * (DP.C / (fl * 1e6)) ** 2 / (4 * math.pi)
    kjy = ae / (2 * K_B) * 1e-26
    out(f"{fl:.1f} MHz: G = {gl:.1f} dBi, HPBW {g.hpbw_deg(fl):.2f} deg, Tsys = {tl:.0f} K, "
        f"Ae = {ae:.1f} m2, {kjy * 1e3:.2f} mK/Jy")
    for B, tau, label in ((5e3, 60, "HI, 1 km/s channel, 1 min"), (5e3, 600, "HI, 1 km/s channel, 10 min")):
        s = tl / math.sqrt(B * tau)
        out(f"  {label:32s} noise {s * 1e3:5.0f} mK  -> galactic plane (~30-80 K) SNR {30 / s:5.0f}; "
            f"high latitude (~2 K) SNR {2 / s:4.0f}")
    for name, S in (("Sun (quiet, ~60 sfu)", 6e5), ("Cas A", 1500.0), ("Cyg A", 1600.0), ("Tau A", 900.0)):
        ta = kjy * S
        snr = ta / (tl / math.sqrt(20e6 * 10))
        out(f"  {name:20s} {S:9.0f} Jy -> T_A {ta:8.2f} K, SNR {snr:9.0f} in 10 s, 20 MHz (pointing calibrators)")
    out(f"  Moon (220 K disk) -> T_A {220 * moon_fill(g, fl):.1f} K")
    # pulsar B0329+54
    S, P_s, W = 0.203, 0.7145, 0.0066
    for hours in (0.5, 2.0):
        snr = (S * kjy / tl) * math.sqrt(2 * hours * 3600 * 20e6) * math.sqrt((P_s - W) / W)
        out(f"  PSR B0329+54 (~0.2 Jy) folded, 20 MHz, 2 pol, {hours:.1f} h: SNR ~{snr:.0f} "
            f"(needs a clean band; RFI is the real limit)")

    # ------------------------------------------------------------------ wind
    out("\n--- Wind, drives, foundation ---")
    A = g.area
    z_el = p.el_axis_height / 1000
    qp = q_peak(p, z_el + p.D / 2000)
    v_pk = math.sqrt(2 * qp / RHO)
    q_op = 0.5 * RHO * (1.5 * p.v_operate) ** 2  # gusts 1.5 x mean during operation
    out(f"survival: peak velocity pressure {qp:.0f} Pa (gust ~{v_pk:.0f} m/s, vb0 {p.vb0:.0f} m/s, z0 {p.terrain_z0} m); "
        f"operation up to {p.v_operate:.0f} m/s mean ({q_op:.0f} Pa with gusts)")

    F_op = p.cf_face * q_op * A
    T_el_op = p.cm_el * q_op * A * p.D / 1000 + g.m_moving * 9.81 * abs(g.cg_el) / 1000
    T_az_op = p.cm_az * q_op * A * p.D / 1000
    Fh_s = p.cf_stow_h * qp * A
    Fv_s = p.cf_stow_v * qp * A
    T_el_s = p.cm_el * qp * A * p.D / 1000
    out(f"operating, dish face-on: drag {F_op / 1e3:.2f} kN, el torque {T_el_op:.0f} Nm, az torque {T_az_op:.0f} Nm")
    out(f"stowed at zenith, survival: horizontal {Fh_s / 1e3:.2f} kN, lift {Fv_s / 1e3:.2f} kN, el torque {T_el_s:.0f} Nm")

    for axis, drv_key, T_op, T_hold, M_tilt in (
            ("az", p.az_drive, T_az_op, p.cm_az * qp * A * p.D / 1000, Fh_s * (p.yoke_height / 1000 + 0.3)),
            ("el", p.el_drive, T_el_op, T_el_s, Fh_s * abs(p.el_drive_x) / 1000 + g.m_moving * 9.81 * abs(p.el_drive_x) / 1000)):
        d = DP.DRIVES[drv_key]
        motor = T_op / (d["ratio"] * p.gearbox_ratio * p.worm_eff * 0.9)
        w = math.radians(p.slew_rate_dps)
        p_mech = T_op * w
        p_el = p_mech / (p.worm_eff * 0.9 * 0.85)
        rpm = p.slew_rate_dps / 360 * 60 * d["ratio"] * p.gearbox_ratio
        out(f"{axis}: {drv_key} (UNVERIFIED: {d['out_torque'] / 1e3:.1f} kNm out, {d['hold_torque'] / 1e3:.1f} kNm hold, "
            f"{d['tilt_moment'] / 1e3:.1f} kNm tilt) | operating torque {T_op:.0f} Nm = {T_op / d['out_torque'] * 100:.0f} % | "
            f"survival {T_hold:.0f} Nm = {T_hold / d['hold_torque'] * 100:.0f} % of hold | tilt {M_tilt:.0f} Nm "
            f"= {M_tilt / d['tilt_moment'] * 100:.0f} %")
        out(f"    servo: {motor:.2f} Nm needed vs {p.servo_rated_torque:.2f} Nm rated "
            f"({motor / p.servo_rated_torque * 100:.0f} %), {rpm:.0f} rpm at {p.slew_rate_dps:.1f} deg/s; "
            f"electrical ~{p_el:.0f} W while slewing against wind")

    # pointing: backlash
    for f_mhz, name in ((p.f_s_mhz, "S"), (p.f_hi_mhz, "L")):
        loss = 12 * (p.backlash_deg / g.hpbw_deg(f_mhz)) ** 2
        out(f"backlash {p.backlash_deg:.2f} deg -> worst-case pointing loss {loss:.2f} dB at {name}-band")

    # foundation: overturning about the footing edge. Design case: stow FAILED, dish face-on at the
    # survival gust (more severe than the stowed case); no lift credit.
    Fh_s = max(Fh_s, p.cf_face * qp * A)
    Fv_s = 0.0
    out(f"foundation design case: stow failed, dish face-on at the survival gust: {Fh_s / 1e3:.2f} kN horizontal")
    L, W_, H = (v / 1000 for v in p.footing)
    m_found = L * W_ * H * p.concrete_density
    pd, pt = p.pedestal_d / 1000, p.pedestal_t / 1000
    ped_len = g.pedestal_top / 1000
    m_ped = math.pi * pd * pt * ped_len * 7850
    m_struct = m_ped + g.az["mass"] + g.el["mass"] + 60.0 + g.m_moving  # +60 kg yoke, cradle, boxes
    F_ped = 1.2 * qp * pd * ped_len
    M_ot = Fh_s * (z_el + H) + F_ped * (ped_len / 2 + H)
    M_res = (m_found + m_struct) * 9.81 * L / 2 - Fv_s * L / 2
    fos = M_res / M_ot
    out(f"foundation {L:.1f} x {W_:.1f} x {H:.1f} m concrete ({m_found:.0f} kg, {L * W_ * H:.2f} m3), structure "
        f"{m_struct:.0f} kg: overturning {M_ot / 1e3:.1f} kNm vs restoring {M_res / 1e3:.1f} kNm -> FoS {fos:.2f} "
        f"({'OK' if fos >= p.fos_overturn else 'TOO SMALL'}, need {p.fos_overturn})")
    # smallest cube-ish footing meeting FoS
    for s in np.arange(0.6, 2.01, 0.05):
        mf = s * s * H * p.concrete_density
        if ((mf + m_struct) * 9.81 * s / 2 - Fv_s * s / 2) / (Fh_s * (z_el + H) + F_ped * (ped_len / 2 + H)) >= p.fos_overturn:
            out(f"    smallest square footing at {H:.1f} m depth for FoS {p.fos_overturn}: {s:.2f} m "
                f"(~{s * s * H:.2f} m3). Also check soil bearing and local frost depth (~0.8 m).")
            break
    out(f"    anchor bolts: design uplift per bolt (4 bolts, pitch circle 300 mm) ~ "
        f"{(M_ot / 0.212 / 2 - (m_struct * 9.81 - Fv_s) / 4) / 1e3:.1f} kN -> 4 x M20 8.8 HDG with nut + plate")

    # ------------------------------------------------------------------ power
    out("\n--- Power at the dish (tracking day) ---")
    loads = {"single-board computer": 8, "SDR (2 ch, active)": 6, "LNAs (2)": 2, "GPSDO": 4,
             "servo drives idle (2)": 4, "5 GHz bridge": 4}
    pw = sum(loads.values())
    out(f"tracking load ~{pw} W ({', '.join(f'{k} {v} W' for k, v in loads.items())})")
    out(f"12 h tracking + 12 h standby at ~8 W: {(pw * 12 + 8 * 12) / 1000:.2f} kWh/day; slewing adds a few Wh "
        f"(worm drives are self-locking: no holding current)")

    with open(os.path.join(OUT, "dish_analysis.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
