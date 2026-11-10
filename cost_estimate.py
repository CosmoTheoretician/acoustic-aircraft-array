#!/usr/bin/env python3
"""
Priced materials list: reads out/bom.csv (written by array_model.py, so quantities follow the
parameters) and adds the electronics that are not part of the mechanical model.

Prices are rough EUR estimates incl. VAT for single-unit / hobby quantities, as of late 2026, from
typical EU web-shop price levels (extrusion and fastener shops, laser-cut services, solar
retailers). They are NOT quotes: every line has a low and a high value, and the spread is the
honest uncertainty. Check current prices before ordering.

Run:  .venv/bin/python cost_estimate.py      (after array_model.py)
Writes out/cost_estimate.csv (custom plates, as modelled) and out/cost_estimate_standard.csv
(standard catalogue plates where they fit, see STANDARD below) and prints both totals.

Standard-parts variant: catalogue 3030 joining plates replace the hub, mid-side tee, splice
and leg-base plates, and DIN 71752 fork heads screwed into the tapped profile cores replace
the custom clevis brackets. Repeatability is kept by match-drilling: at first assembly each
standard plate is clamped in place and the dowel holes are drilled and reamed through plate
and profile with the jig, then the plate is numbered to that position. The corner gussets
stay custom: the 45-degree arm, both side halves, the outrigger and the leg meet there, and
no catalogue plate takes that.
"""

import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SPARES = 1.10  # fasteners: +10 %

# ---- unit prices, EUR (low, high) ----
PROFILE_PER_M = {"3030": (7.5, 11.0), "4040": (11.0, 16.0)}  # B-type slot 8/10, anodised
CUT_FEE = (0.5, 1.2)  # per cut to length (most extrusion shops cut to +-0.5 mm)


def plate_price(area_mm2):  # laser / waterjet, 6 mm 5083, incl. material, single parts
    return 3.0 + 0.00035 * area_mm2, 6.0 + 0.0006 * area_mm2


# regex on the BOM item -> (low, high) per unit, group, comment
PRICES = [
    (r"^Corner bracket", (0.8, 1.8), "frame", "cast Zn/Al, 28x28"),
    (r"^Ground screw .*head plate", (18, 30), "frame", "galvanised, 650 mm"),
    (r"^Ground screw .*pole sleeve", (25, 40), "frame", ""),
    (r"^Clevis bracket", (5, 12), "frame", "laser-cut + bent 4 mm stainless (custom)"),
    (r"^Al plate .*box mounting", (15, 30), "frame", ""),
    (r"^Weatherproof enclosure", (80, 160), "power", "polycarbonate IP66 (Fibox/Spelsberg class)"),
    (r"^PV panel", (80, 130), "power", "200 W mono"),
    (r"^MPPT", (90, 130), "power", "e.g. Victron SmartSolar 100/20"),
    (r"^LiFePO4", (250, 400), "power", "with low-temperature charge protection"),
    (r"^PV cable", (30, 60), "power", ""),
    (r"^Aluminium or galvanised tube", (45, 90), "pole", "incl. internal sleeve"),
    (r"^U-bolt pole clamp", (6, 12), "pole", ""),
    (r"^Ultrasonic 2D wind sensor", (250, 1400), "sensors", "generic RS485 ultrasonic ... Gill WindSonic"),
    (r"^TP-Link CPE210", (40, 55), "network", ""),
    (r"^Hose clamp", (1.5, 3), "pole", ""),
    (r"^Sky camera", (60, 250), "sensors", "e.g. Raspberry Pi HQ camera + fisheye + housing"),
    (r"^Armoured / steel-braided cable conduit", (90, 180), "power", "~25-30 m"),
    (r"^Foam windscreen", (8, 25), "microphones", "foam + furry cover"),
    (r"^XLR cable", (15, 30), "microphones", "PUR outdoor, 5-8 m"),
    # fasteners (per piece)
    (r"^Clevis pin", (1.0, 2.0), "fasteners", ""),
    (r"^Dowel pin", (0.25, 0.50), "fasteners", ""),
    (r"^M10 threaded rod", (1.2, 2.2), "fasteners", "per stud set"),
    (r"^M3x16", (0.15, 0.30), "fasteners", ""),
    (r"^M4x30", (0.20, 0.40), "fasteners", ""),
    (r"^M5x16", (0.15, 0.30), "fasteners", ""),
    (r"^M6 nyloc", (0.08, 0.15), "fasteners", ""),
    (r"^M6 washer", (0.03, 0.06), "fasteners", ""),
    (r"^M6x\d+ button head", (0.12, 0.25), "fasteners", ""),
    (r"^M6x50", (0.35, 0.60), "fasteners", ""),
    (r"^M8x16", (0.25, 0.50), "fasteners", ""),
    (r"^PV mid/end clamp", (2.5, 5.0), "fasteners", ""),
    (r"^T-nut", (0.30, 0.60), "fasteners", "stainless hammer nuts"),
]

PRINTED_EUR = {"ASA": (1.5, 3.0), "TPU": (0.5, 1.5)}  # filament per part, ~25-35 EUR/kg

# not in the mechanical model: (item, qty, low, high, group, comment)
EXTRAS = [
    ("Behringer ECM8000 measurement mic (5 + optional 6th)", 6, 25, 40, "microphones", ""),
    ("Audio interface, 8 phantom-powered inputs, one sample clock (e.g. Behringer UMC1820)", 1, 230, 320,
     "electronics", "all mics on ONE converter clock; TDOA needs it"),
    ("12 V -> 230 V pure-sine inverter 150 W (for a mains-powered interface)", 1, 50, 100, "electronics",
     "skip if the interface runs on 12 V DC"),
    ("Raspberry Pi 5 8 GB + case + NVMe SSD + 12 V->5 V 5 A converter", 1, 140, 210, "electronics", ""),
    ("ADS-B receiver: RTL-SDR v4 + 1090 MHz filter + antenna", 1, 70, 130, "electronics", "validation truth"),
    ("TP-Link CPE210, far end of the link (at the house) + PoE injector", 1, 40, 55, "network", "a link needs 2"),
    ("Outdoor Ethernet cable (CAT6 PE) for the CPE210, ~10 m + glands", 1, 15, 30, "network", ""),
    ("Box internals: DIN rail, fuses, terminals, DC-DC 12->24 V passive PoE", 1, 40, 80, "power", ""),
    ("Earth rod + 16 mm2 cable for the pole (tallest point in the garden)", 1, 30, 60, "pole", ""),
    ("Reamers 6 H7 (dowels) + 8 H8 (pins), M8 tap, drill-jig bushings", 1, 50, 100, "tools", "DIY machining"),
    ("Core tapping of profile ends (if done by the extrusion shop)", 30, 1.0, 2.0, "tools", "or DIY with the tap"),
]


# plate key -> (replacement, low, high EUR per plate); keys not listed stay custom
STANDARD = {
    "hub_top": ("Cross joining plate 3030, 4-arm (catalogue), match-drilled for dowels + mast", 6, 12),
    "hub_bottom": ("Cross joining plate 3030, 4-arm (catalogue), match-drilled for dowels", 6, 12),
    "midside_tee": ("T joining plate 3030 (catalogue), match-drilled for dowels", 3, 6),
    "splice": ("Straight joining plate 3030, >=120 mm, 4 holes (catalogue), match-drilled", 2.5, 4.5),
    "leg_base": ("Galvanised steel base plate ~120 x 120 x 6, drilled to the screw-head studs", 4, 8),
}
# custom clevis bracket -> fork head on the member end + catalogue bracket drilled 8.2 mm as the lug
CLEVIS_STD = ("Fork head DIN 71752 G8x16 stainless + M8 jam nut (into the tapped profile core), "
              "lug = 3030 corner bracket drilled 8.2 mm", 4, 8)


def build(bom, prof, standard):
    rows_out = []
    for r in bom:
        cat, item, spec, q = r["category"], r["item"], r["spec"], r["qty"]
        if not q:
            continue
        q = int(q)
        if cat == "profile":
            L = float(re.search(r"cut ([\d.]+) mm", spec).group(1)) / 1000
            pm = PROFILE_PER_M[prof]
            lo = q * (L * pm[0] + CUT_FEE[0])
            hi = q * (L * pm[1] + CUT_FEE[1])
            rows_out.append(("frame", item, f"{q} x {L * 1000:.0f} mm", q, lo, hi, "incl. cutting"))
        elif cat == "plate" and standard and item in STANDARD:
            name, lo, hi = STANDARD[item]
            rows_out.append(("frame", name, f"replaces custom {item}", q, q * lo, q * hi, "dowels match-drilled"))
        elif cat == "bought" and standard and item.startswith("Clevis bracket"):
            name, lo, hi = CLEVIS_STD
            rows_out.append(("frame", name, "replaces custom clevis", q, q * lo, q * hi,
                             "set length once, lock, paint-mark"))
        elif cat == "plate":
            w, h = map(float, re.search(r"envelope (\d+) x (\d+)", spec).groups())
            lo, hi = plate_price(w * h)
            rows_out.append(("frame", f"Plate {item} (6 mm Al, laser/waterjet)", spec.split(",")[2].strip(), q,
                             q * lo, q * hi, "reamed dowel holes DIY"))
        elif cat == "printed":
            lo, hi = PRINTED_EUR["TPU" if "TPU" in spec else "ASA"]
            rows_out.append(("printed", item, spec, q, q * lo, q * hi, "filament only"))
        else:
            qq = q * SPARES if cat == "fastener" else q
            for pat, (lo, hi), grp, com in PRICES:
                if re.search(pat, item):
                    rows_out.append((grp, item, spec, round(qq), qq * lo, qq * hi, com))
                    break
            else:
                rows_out.append(("unpriced", item, spec, q, 0.0, 0.0, "NO PRICE"))
    for item, q, lo, hi, grp, com in EXTRAS:
        rows_out.append((grp, item, "", q, q * lo, q * hi, com))
    return rows_out


ORDER = ["frame", "fasteners", "printed", "pole", "microphones", "sensors", "electronics", "network",
         "power", "tools", "unpriced"]


def write(rows_out, path):
    order = ORDER
    rows_out.sort(key=lambda r: order.index(r[0]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "item", "spec", "qty", "eur_low", "eur_high", "comment"])
        for r in rows_out:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.0f}", f"{r[5]:.0f}", r[6]])
        tl, th = sum(r[4] for r in rows_out), sum(r[5] for r in rows_out)
        w.writerow(["TOTAL", "", "", "", f"{tl:.0f}", f"{th:.0f}", ""])
    return {g: (sum(r[4] for r in rows_out if r[0] == g), sum(r[5] for r in rows_out if r[0] == g))
            for g in order}, (tl, th)


def main():
    with open(os.path.join(OUT, "bom.csv")) as f:
        bom = list(csv.DictReader(f))
    prof = next((r["item"].split()[0] for r in bom if r["category"] == "profile"), "3030")
    a, ta = write(build(bom, prof, False), os.path.join(OUT, "cost_estimate.csv"))
    b, tb = write(build(bom, prof, True), os.path.join(OUT, "cost_estimate_standard.csv"))
    print(f"{'EUR':12s} {'custom plates':>15s} {'standard plates':>17s}")
    for g in ORDER:
        if a[g] != (0, 0) or b[g] != (0, 0):
            print(f"{g:12s} {a[g][0]:7.0f}-{a[g][1]:<7.0f} {b[g][0]:9.0f}-{b[g][1]:<7.0f}")
    print(f"{'TOTAL':12s} {ta[0]:7.0f}-{ta[1]:<7.0f} {tb[0]:9.0f}-{tb[1]:<7.0f}")
    print(f"saving: {ta[0] - tb[0]:.0f}-{ta[1] - tb[1]:.0f} EUR")


if __name__ == "__main__":
    main()
