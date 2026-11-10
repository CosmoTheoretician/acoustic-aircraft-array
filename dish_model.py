#!/usr/bin/env python3
"""
CadQuery model of the 3.5 m dish on its elevation-over-azimuth positioner.

Layout (dish frame: origin at the pedestal axis at ground level, x = east, y = north, z = up):
  - concrete footing, galvanised base plate on 4 x M20 anchors, 219.1 mm steel pedestal tube;
  - azimuth slewing drive (worm, self-locking) on the pedestal flange, servo + planetary on its
    worm; the turntable on the drive's output ring carries everything above;
  - yoke: one upright plate (with two gussets) outboard of the elevation slewing drive, whose
    axis is horizontal and runs BEHIND the reflector backframe; its output plate carries the
    el beam, the backframe hub and the counterweight arm;
  - commercial mesh reflector (shown as envelope: shell + 24 ribs + hub), quadripod to a
    quick-swap feed plate at the focus (2 dowel pins + 4 bolts: S-band and L-band feeds swap
    without re-focusing), LNA box behind the feed;
  - electronics box (SDR, GPSDO, computer, servo drives) on the yoke, GPS antenna on top:
    only 48 V power and Ethernet go through the azimuth cable wrap.

Run:  .venv/bin/python dish_model.py [--az 150 --el 35] [--no-render]
Writes out/dish/dish_assembly.step, out/dish/dish_bom.csv, out/dish/render/*.png.
The script also sweeps elevation 0..90 deg and checks every moving part against the yoke,
drives and pedestal (boolean intersection volumes), and the rim's ground clearance.
"""

import argparse
import collections
import csv
import math
import os
import shutil
import subprocess

import cadquery as cq

import dish_params as DP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "dish")

C_CONC = (0.72, 0.72, 0.70, 1)
C_GALV = (0.62, 0.64, 0.66, 1)
C_DRIVE = (0.25, 0.30, 0.35, 1)
C_MOTOR = (0.10, 0.10, 0.12, 1)
C_MESH = (0.80, 0.82, 0.85, 0.55)
C_RIB = (0.70, 0.72, 0.75, 1)
C_FEED = (0.85, 0.65, 0.20, 1)
C_BOX = (0.82, 0.82, 0.78, 1)
C_CW = (0.35, 0.35, 0.38, 1)


def V(x, y, z):
    return cq.Vector(float(x), float(y), float(z))


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def cyl(r, h, p, d):
    return cq.Solid.makeCylinder(r, h, V(*p), V(*d))


def tube_between(a, b, r):
    a, b = cq.Vector(*a), cq.Vector(*b)
    d = b - a
    return cq.Solid.makeCylinder(r, d.Length, a, d.normalized())


class DishModel:
    def __init__(self, p: DP.DishParams, az=None, el=None):
        self.p = p
        self.g = DP.DishGeometry(p)
        self.az = p.park_az if az is None else az
        self.el = p.park_el if el is None else el
        self.fixed, self.azp, self.elp = [], [], []  # (name, shape, colour) in their own frames
        self.buy = collections.OrderedDict()

    # ---------------------------------------------------------------- frames
    def to_world_az(self, s, az=None):
        return s.rotate(V(0, 0, 0), V(0, 0, 1), -(self.az if az is None else az))

    def to_world_el(self, s, el=None, az=None):
        s = s.rotate(V(0, 0, 0), V(1, 0, 0), self.el if el is None else el)
        s = s.translate(V(0, 0, self.p.el_axis_height))
        return self.to_world_az(s, az)

    def parts_world(self, az=None, el=None):
        out = [(n, s, c) for n, s, c in self.fixed]
        out += [(n, self.to_world_az(s, az), c) for n, s, c in self.azp]
        out += [(n, self.to_world_el(s, el, az), c) for n, s, c in self.elp]
        return out

    def add_buy(self, item, qty, lo, hi, note=""):
        q0, l0, h0, n0 = self.buy.get(item, (0, lo, hi, note))
        self.buy[item] = (q0 + qty, lo, hi, note)

    # ---------------------------------------------------------------- build
    def build(self):
        p, g = self.p, self.g
        az_d, el_d = g.az, g.el
        Lf, Wf, Hf = p.footing
        # ---- fixed: footing, base plate, pedestal, az drive housing + motor ----
        self.fixed.append(("footing", box(-Lf / 2, Lf / 2, -Wf / 2, Wf / 2, 50 - Hf, 50), C_CONC))
        self.fixed.append(("base_plate", box(-200, 200, -200, 200, 50, 75), C_GALV))
        for sx in (1, -1):
            for sy in (1, -1):
                self.fixed.append(("anchor_nut", cyl(16, 35, (sx * 106, sy * 106, 75), (0, 0, 1)), C_GALV))
        z_flange = g.pedestal_top - 20
        rp = p.pedestal_d / 2
        ped = cyl(rp, z_flange - 75, (0, 0, 75), (0, 0, 1)).cut(
            cyl(rp - p.pedestal_t, z_flange - 73, (0, 0, 74), (0, 0, 1)))
        self.fixed.append(("pedestal_tube", ped, C_GALV))
        self.fixed.append(("pedestal_flange", cyl(200, 20, (0, 0, z_flange), (0, 0, 1)), C_GALV))
        for k in range(4):  # base gussets
            gs = box(rp - 2, 200, -6, 6, 75, 75 + 180).cut(
                box(rp - 3, 201, -7, 7, 75, 255).rotate(V(rp, 0, 255), V(rp, 1, 255), 0))
            tri = cq.Workplane("XZ").polyline([(rp - 1, 75), (195, 75), (rp - 1, 255)]).close().extrude(6, both=True).val()
            self.fixed.append(("base_gusset", tri.rotate(V(0, 0, 0), V(0, 0, 1), 45 + 90 * k), C_GALV))
        za0 = g.pedestal_top
        self.fixed.append(("az_drive", cyl(az_d["housing_d"] / 2, az_d["height"], (0, 0, za0), (0, 0, 1)), C_DRIVE))
        r_az = az_d["housing_d"] / 2
        self.fixed.append(("az_worm_housing", box(r_az - 40, r_az + 60, -170, 170, za0 + 15, za0 + 115), C_DRIVE))
        self.fixed.append(("az_gearbox", cyl(45, 110, (r_az + 10, -170, za0 + 65), (0, -1, 0)), C_MOTOR))
        self.fixed.append(("az_servo", cyl(42, 160, (r_az + 10, -280, za0 + 65), (0, -1, 0)), C_MOTOR))
        self.add_buy(f"Slewing drive {p.az_drive} (worm, self-locking, IP65), azimuth", 1, 500, 900,
                     "UNVERIFIED ratings in dish_params.DRIVES")
        self.add_buy(f"Slewing drive {p.el_drive} (worm, self-locking, IP65), elevation", 1, 350, 650, "")
        self.add_buy("Integrated 48 V servo 400 W with encoder + driver, IP65", 2, 250, 450, "")
        self.add_buy(f"Planetary gearbox {p.gearbox_ratio:.0f}:1 (servo flange to worm shaft adapter)", 2, 120, 250, "")
        self.add_buy("Absolute encoder 17-bit, IP67, on each drive output (belt 1:1)", 2, 150, 350,
                     "reads the output: backlash and worm wear don't affect pointing knowledge")
        self.add_buy("Limit switches (2 per axis) + hardware E-stop + brake resistor", 1, 60, 150, "")
        self.add_buy(f"Pedestal tube {p.pedestal_d} x {p.pedestal_t} S355, {g.pedestal_top / 1000:.2f} m, "
                     f"with base plate 400x400x25, top flange, 4 gussets, hot-dip galvanised", 1, 350, 700,
                     "fabricated")
        self.add_buy(f"Concrete C25/30 {Lf * Wf * Hf * 1e-9:.2f} m3 + rebar cage + 4 x M20 8.8 HDG anchors with "
                     f"template", 1, 300, 550, "frost-proof depth")

        # ---- az-rotating: turntable, yoke upright + gussets, el drive housing, box ----
        z_az = g.z_az
        x_up1 = p.el_drive_x - el_d["height"] / 2
        x_up0 = x_up1 - 20
        self.azp.append(("turntable", cyl(200, 20, (0, 0, z_az), (0, 0, 1)), C_GALV))
        z_top = p.el_axis_height + el_d["housing_d"] / 2 + 20
        self.azp.append(("yoke_upright", box(x_up0, x_up1, -170, 170, z_az + 20, z_top), C_GALV))
        for sy in (1, -1):
            tri = cq.Workplane("XZ").polyline([(x_up1, z_az + 20), (x_up1 + 60, z_az + 20),
                                               (x_up1, z_az + 420)]).close().extrude(7.5, both=True).val()
            self.azp.append(("yoke_gusset", tri.translate(V(0, sy * 162.5, 0)), C_GALV))
        h = p.el_axis_height
        self.azp.append(("el_drive", cyl(el_d["housing_d"] / 2, el_d["height"], (x_up1, 0, h), (1, 0, 0)), C_DRIVE))
        r_el = el_d["housing_d"] / 2
        self.azp.append(("el_worm_housing", box(x_up1 + 10, x_up1 + el_d["height"] - 10, -160, 160,
                                                h - r_el - 50, h - r_el + 40), C_DRIVE))
        self.azp.append(("el_gearbox", cyl(45, 110, (p.el_drive_x, -160, h - r_el - 5), (0, -1, 0)), C_MOTOR))
        self.azp.append(("el_servo", cyl(42, 160, (p.el_drive_x, -270, h - r_el - 5), (0, -1, 0)), C_MOTOR))
        bx0 = x_up0 - 220
        self.azp.append(("electronics_box", box(bx0, x_up0 - 5, -200, 200, h - 760, h - 260), C_BOX))
        self.azp.append(("gps_antenna", cyl(35, 25, ((bx0 + x_up0) / 2, 0, h - 260), (0, 0, 1)), C_MOTOR))
        self.add_buy("Yoke: upright 340x20 S355 + 2 gussets + turntable 400x20, laser-cut, galvanised", 1, 300, 600, "")
        self.add_buy("Electronics enclosure 400x500x220 IP66 on the yoke, vent + glands", 1, 100, 200, "")
        self.add_buy("Cable wrap / drag chain for +-270 deg azimuth (48 V, Ethernet, E-stop loop)", 1, 80, 150, "")

        # ---- el-rotating (boresight +y, el axis = x, origin on the axis) ----
        a = p.axis_behind_vertex
        f = g.f
        R = p.D / 2
        # reflector shell (mesh envelope) + ribs + hub
        n = 48
        rs = [R * i / n for i in range(n + 1)]
        outer = [(r, a + r * r / (4 * f)) for r in rs]
        inner = [(r, y + 3) for r, y in reversed(outer)]
        shell = cq.Workplane("XY").polyline(outer + inner).close().revolve(360, (0, 0, 0), (0, 1, 0)).val()
        self.elp.append(("reflector_mesh", shell, C_MESH))
        r0 = p.hub_d / 2
        rib_pts = [(r, a + r * r / (4 * f)) for r in [r0 + (R - r0) * i / 20 for i in range(21)]]
        hgt = [80 - 55 * i / 20 for i in range(21)]
        rib2d = rib_pts + [(r, y - hh) for (r, y), hh in reversed(list(zip(rib_pts, hgt)))]
        rib = cq.Workplane("XY").polyline(rib2d).close().extrude(4, both=True).val()
        for k in range(24):
            self.elp.append(("rib", rib.rotate(V(0, 0, 0), V(0, 1, 0), 15 * k), C_RIB))
        self.elp.append(("backframe_hub", cyl(r0, p.backframe_depth, (0, a - p.backframe_depth, 0), (0, 1, 0)), C_RIB))
        self.add_buy(f"Aluminium mesh reflector kit {p.D / 1000:.1f} m, f/D {p.f_over_D}, 24 ribs, hub, mesh "
                     f"<= {p.mesh_opening:.0f} mm", 1, 2500, 5000, "get a quote; check 3.5 m is offered")
        # el adapter, beam, posts to the hub plate, counterweight arm + plates
        x_out = x_up1 + el_d["height"]  # drive output face
        self.elp.append(("el_output_plate", cyl(150, 20, (x_out, 0, 0), (1, 0, 0)), C_GALV))
        self.elp.append(("el_beam", box(x_out + 20, 160, -50, 50, -50, 50), C_GALV))
        y_hp = a - p.backframe_depth
        self.elp.append(("hub_plate", box(-200, 200, y_hp - 15, y_hp, -200, 200), C_GALV))
        for sx in (x_out + 60, 110):  # posts stay inboard of the el drive output face
            self.elp.append(("hub_post", box(sx - 40, sx + 40, 50, y_hp - 15, -40, 40), C_GALV))
        cw_t = g.cw_mass / (7850 * 0.24 * 0.34) * 1000
        self.elp.append(("cw_arm", box(-40, 40, -p.cw_arm - cw_t / 2 - 20, -50, -40, 40), C_GALV))
        self.elp.append(("counterweight", box(-120, 120, -p.cw_arm - cw_t / 2, -p.cw_arm + cw_t / 2, -170, 170), C_CW))
        self.add_buy("El cradle: output plate, beam 100x100x5, hub plate, posts, cw arm 80x80x4, galvanised",
                     1, 200, 400, "")
        self.add_buy(f"Counterweight: steel plates 240x340, {g.cw_mass:.0f} kg total (~{cw_t:.0f} mm stack), "
                     f"trim on site", 1, 80, 150, "balance with feed fitted")
        # quadripod + quick-swap feed plate + S-band feed + LNA box
        fd, fl = p.s_feed
        y_focus = a + f
        y_fp = y_focus + fl
        self.elp.append(("feed_plate_fixed", box(-110, 110, y_fp, y_fp + 10, -110, 110), C_GALV))
        self.elp.append(("feed_plate_swap", box(-90, 90, y_fp - 10, y_fp, -90, 90), C_FEED))
        self.elp.append(("feed_s_band", cyl(fd / 2, fl - 10, (0, y_focus, 0), (0, 1, 0)), C_FEED))
        self.elp.append(("feed_s_mouth", cq.Solid.makeCone(fd / 2 + 25, fd / 2, 40, V(0, y_focus, 0), V(0, 1, 0)), C_FEED))
        self.elp.append(("lna_box", box(-80, 80, y_fp + 10, y_fp + 70, -50, 50), C_BOX))
        yq = a + p.quadripod_r ** 2 / (4 * f)
        for k in range(4):
            ang = math.radians(45 + 90 * k)
            foot = (p.quadripod_r * math.cos(ang), yq, p.quadripod_r * math.sin(ang))
            top = (95 * math.cos(ang) * math.sqrt(2), y_fp + 5, 95 * math.sin(ang) * math.sqrt(2))
            self.elp.append(("quadripod_leg", tube_between(foot, top, 15), C_RIB))
        self.add_buy("Quadripod: 4 x Al tube 30x2, 2.0 m, end fittings to ribs; fixed feed plate 220x220x10", 1, 120, 250, "")
        self.add_buy("Quick-swap feed plates (one per feed) with 2 x 6 mm dowels + 4 x M6 thumb bolts", 2, 20, 50, "")
        self.add_buy(f"S-band dual-CP septum feed for f/D {p.f_over_D}, 2.2-2.3 GHz", 1, 250, 600, "or DIY")
        self.add_buy(f"L-band 1420 MHz circular waveguide feed with choke, f/D {p.f_over_D}", 1, 100, 300, "DIY feasible")
        # RF + electronics
        self.add_buy("S-band LNA 0.5 dB NF, 30 dB, 2.2-2.3 GHz", 2, 100, 250, "one per CP port")
        self.add_buy("S-band cavity band-pass filter 2.2-2.3 GHz (rejects 2.4 GHz Wi-Fi)", 2, 60, 150, "after the LNA")
        self.add_buy("L-band LNA + 1420 MHz filter (hydrogen line)", 1, 40, 90, "")
        self.add_buy("Switched noise source for calibration (+ coupler)", 1, 60, 150, "")
        self.add_buy("2-channel coherent SDR, 70 MHz-6 GHz, ext. 10 MHz + PPS (USRP B210 class)", 1, 1500, 2300, "")
        self.add_buy("GPSDO with OCXO: 10 MHz + PPS, ADEV <= 1e-11 at 1 s", 1, 250, 600,
                     "the receiver's frequency reference: the core of the Doppler accuracy")
        self.add_buy("GPS antenna (active, timing grade)", 1, 20, 60, "")
        self.add_buy("Computer at the dish (Pi 5 8 GB / mini PC) + NVMe for raw IQ", 1, 150, 300, "")
        self.add_buy("Coax LMR-240 feed->box 2 x 4 m, connectors, lightning arrestors", 1, 80, 150, "")
        self.add_buy("DC-DC 12 V -> 48 V 600 W at the battery + 48 V buried cable", 1, 150, 300, "")
        self.add_buy("5 GHz bridge pair (e.g. TP-Link CPE510) replacing the CPE210", 1, 90, 130, "")
        self.add_buy("Earth rod + 16 mm2 bonding of pedestal, yoke and coax shields", 1, 40, 80, "")

    # ---------------------------------------------------------------- checks
    def sweep_check(self, az=None):
        fixed_like = [s for n, s, c in self.fixed] + [self.to_world_az(s, az) for n, s, c in self.azp]
        fixed_c = cq.Compound.makeCompound(fixed_like)
        worst = 0.0
        min_z = 1e9
        for el in range(0, 91, 10):
            moving = cq.Compound.makeCompound([self.to_world_el(s, el, az) for n, s, c in self.elp])
            v = moving.intersect(fixed_c).Volume()
            if v > 1:  # name the offending pairs
                for n, s, c in self.elp:
                    w = self.to_world_el(s, el, az)
                    for fn, fs in zip([n2 for n2, _, _ in self.fixed] + [n2 for n2, _, _ in self.azp], fixed_like):
                        vv = w.intersect(fs).Volume()
                        if vv > 1:
                            print(f"  COLLISION el {el}: {n} x {fn}: {vv:.0f} mm3")
            worst = max(worst, v)
            bb = moving.BoundingBox()
            min_z = min(min_z, bb.zmin)
        return worst, min_z

    # ---------------------------------------------------------------- outputs
    def export_step(self, path):
        assy = cq.Assembly(name="dish")
        names = collections.Counter()
        for n, s, c in self.parts_world():
            names[n] += 1
            assy.add(s, name=f"{n}_{names[n]}", color=cq.Color(*c))
        assy.export(path)

    def export_bom(self, path):
        tl = th = 0.0
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item", "qty", "eur_low", "eur_high", "note"])
            for item, (q, lo, hi, note) in self.buy.items():
                w.writerow([item, q, f"{q * lo:.0f}", f"{q * hi:.0f}", note])
                tl += q * lo
                th += q * hi
            w.writerow(["TOTAL", "", f"{tl:.0f}", f"{th:.0f}", "rough EUR estimates incl. VAT, not quotes"])
        return tl, th


def render(model, outdir):
    exe = shutil.which("openscad")
    if not exe:
        print("openscad not found; skipping renders")
        return []
    mesh = os.path.join(outdir, "mesh")
    os.makedirs(mesh, exist_ok=True)
    for f in os.listdir(outdir):
        if f.endswith(".png"):
            os.remove(os.path.join(outdir, f))
    pngs = []
    p = model.p
    for pose, (az, el) in {"track": (model.az, model.el), "stow": (model.az, 90.0)}.items():
        groups = collections.defaultdict(list)
        for n, s, c in model.parts_world(az, el):
            groups[c].append(s)
        files = []
        for i, (c, shapes) in enumerate(groups.items()):
            fn = os.path.join(mesh, f"{pose}_{i}.stl")
            cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound(shapes)), fn,
                                tolerance=0.5, angularTolerance=0.2)
            files.append((fn, c))
        scad = os.path.join(mesh, f"{pose}.scad")
        with open(scad, "w") as f:
            f.write("color([0.55,0.65,0.45,1]) translate([-4000,-4000,-2]) cube([8000,8000,2]);\n")
            for fn, c in files:
                f.write(f'color([{c[0]},{c[1]},{c[2]},{c[3]}]) import("{fn}");\n')
        hz = p.el_axis_height
        views = {"iso": (f"7500,-9000,{hz + 4200:.0f},0,0,{hz:.0f}", "1400,1300", "p"),
                 "side": (f"-14000,0,{hz:.0f},0,0,{hz:.0f}", "1300,1200", "o")}
        if pose == "track":
            views["positioner_closeup"] = (f"-2600,-2400,{hz + 600:.0f},-100,0,{hz - 300:.0f}", "1200,1100", "p")
        for name, v in views.items():
            if v is None:
                continue
            cam, size, proj = v
            png = os.path.join(outdir, f"{pose}_{name}.png")
            cmd = [exe, "-o", png, f"--imgsize={size}", f"--camera={cam}", "--colorscheme=Tomorrow",
                   f"--projection={proj}", scad]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                pngs.append(png)
            else:
                print("openscad failed:", r.stderr[-300:])
    return pngs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=float, default=None)
    ap.add_argument("--el", type=float, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    p = DP.DEFAULT
    m = DishModel(p, args.az, args.el)
    m.build()
    os.makedirs(OUT, exist_ok=True)
    m.export_step(os.path.join(OUT, "dish_assembly.step"))
    tl, th = m.export_bom(os.path.join(OUT, "dish_bom.csv"))
    g = m.g
    print(f"parts: {len(m.fixed) + len(m.azp) + len(m.elp)}; pose az {m.az:.0f} el {m.el:.0f}")
    print(f"dish + feed {g.m_dish:.0f} kg, counterweight {g.cw_mass:.0f} kg at {p.cw_arm:.0f} mm, "
          f"moving {g.m_moving:.0f} kg")
    print(f"estimated cost {tl:.0f}-{th:.0f} EUR")
    if not args.no_check:
        for az in (0.0, 90.0):
            v, zmin = m.sweep_check(az)
            print(f"el sweep 0..90 at az {az:.0f}: max overlap moving vs fixed {v:.0f} mm3, "
                  f"lowest moving point {zmin:.0f} mm above ground")
    if not args.no_render:
        pngs = render(m, os.path.join(OUT, "render"))
        print("renders:", [os.path.basename(x) for x in pngs])


if __name__ == "__main__":
    main()
