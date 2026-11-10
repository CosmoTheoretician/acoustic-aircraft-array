#!/usr/bin/env python3
"""
Parametric CAD model of the acoustic aircraft-tracking array (CadQuery).

All dimensions come from array_params.Params. Override any of them on the command line:
    .venv/bin/python array_model.py --set square_side=2500 --set profile=4040
Outputs (in out/):
    array_assembly.step      full assembly, named + coloured parts
    stl/*.stl                every 3D-printed part, in print orientation
    plates/*.dxf             flat aluminium plates for laser / waterjet cutting
    bom.csv                  cut list + bill of materials
    mic_positions.json/.csv  mic capsule coordinates (from the same Params)
    render/*.png             preview images (needs `openscad` on PATH)

Design notes (why it is built this way):
- Elevated: the base frame stands `base_height` (2 m) up on four vertical legs with knee
  braces, on screw-in ground anchors with levelling studs. Out of reach of deer, dogs,
  foxes and boar, mowable underneath, no concrete, removable for winter.
- Base = 4 diagonal arms from a central hub to 4 corner nodes + a perimeter square
  (each side in two halves joined at a mid-side node). Base mics sit at the ends of
  "outriggers" that continue the diagonals 450 mm past the corner nodes, so no strut,
  leg or brace is near a capsule.
- Mast on the hub, 4 struts from the corner nodes to the mast top (pyramid edges). Struts
  and knee braces end in clevis pins: every triangle is closed by fixed, drilled lengths.
- Repeatability: every profile has a jig-drilled H7 through-hole 20 mm from each joint
  end; plates carry matching reamed holes and dowel pins. Bolts + T-nuts only clamp.
- All mics free-field, pointing up, in identical printed holders (TPU isolation sleeve,
  index collar for the axial position, cavity for the XLR plug, side cable exit + drain).
  Base-mic holders bolt flat onto the outrigger end; the apex holder slides over the mast.
- Solar: on a separate ground stand `solar_gap` south of the array, facing true south (-y),
  tilted 60 deg. Not on the array frame: the panels are a ~2 m2 sail that would rock the
  frame (moving the mics) and a large reflector next to the capsules. The array is north of
  the stand, so it never shades it; the stand is low (top edge below the mics), and its glass
  faces away from the array, so sound reflected off it goes up and south, away from the mics.
- Wind sensor (ultrasonic: no moving parts, so no mechanical noise near the mics) on a
  pole on the west mid-side, ~0.9 m above the apex mic, with the camera and a TP-Link CPE210
  Wi-Fi bridge pointing the same way as the solar panels (true south), on a printed saddle.
"""

import argparse
import collections
import math
import os
import shutil
import subprocess
from dataclasses import fields, replace

import cadquery as cq
import numpy as np
from scipy.spatial import ConvexHull

import array_params as AP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# Colours (r, g, b, a)
C_ALU = (0.78, 0.79, 0.82, 1)
C_PLATE = (0.45, 0.47, 0.50, 1)
C_STEEL = (0.60, 0.62, 0.66, 1)
C_PRINT = (0.95, 0.45, 0.10, 1)  # ASA/PETG parts
C_TPU = (0.12, 0.12, 0.12, 1)
C_MIC = (0.30, 0.30, 0.33, 1)
C_WIND = (0.55, 0.55, 0.60, 0.25)  # wind-screen keep-out (not a part)
C_GROUND = (0.25, 0.25, 0.25, 1)
C_BOX = (0.80, 0.82, 0.78, 1)
C_POLE = (0.35, 0.36, 0.38, 1)
C_PV = (0.10, 0.16, 0.35, 1)
C_SENSOR = (0.92, 0.92, 0.92, 1)

Z = np.array([0.0, 0.0, 1.0])
S2 = 1 / math.sqrt(2)


def V(*a):
    return np.array(a, dtype=float)


def nrm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def loc(origin, xdir, normal):
    """Location whose local XY plane sits at origin with local x=xdir, local z=normal."""
    n = nrm(normal)
    x = np.asarray(xdir, float)
    x = nrm(x - n * np.dot(x, n))
    return cq.Location(cq.Plane(origin=tuple(map(float, origin)), xDir=tuple(x), normal=tuple(n)))


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def cyl(r, h, pnt, d):
    return cq.Solid.makeCylinder(r, h, cq.Vector(*map(float, pnt)), cq.Vector(*map(float, d)))


def rot2(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


class Model:
    def __init__(self, p: AP.Params):
        self.p = p
        self.g = AP.derive(p)
        self.P = self.g.P
        self.pr = self.g.prof
        self.parts = []  # (name, shape, colour)
        self.profiles = []  # dict(role, length, note)
        self.plate_defs = {}
        self.plate_count = collections.Counter()
        self.hw = collections.Counter()
        self.printed = collections.Counter()
        self.buy = collections.Counter()
        self._section = None
        self.names = collections.Counter()
        self.checks = {}

    # ------------------------------------------------------------------ basics
    def add(self, name, shape, colour):
        self.names[name] += 1
        self.parts.append((f"{name}_{self.names[name]}", shape, colour))

    @property
    def thr(self):
        return self.pr["thread"]

    def hw_tbolt(self, n=1, length=None):
        length = length or (12 if self.thr == "M6" else 16)
        self.hw[f"{self.thr}x{length} button head ISO 7380, A4 stainless"] += n
        self.hw[f"T-nut {self.thr} for slot {self.pr['slot']:.0f} (hammer or drop-in), A2/A4 stainless"] += n

    def hw_thru(self, n=1):
        self.hw[f"{self.thr}x20 button head ISO 7380, A4 stainless"] += n
        self.hw[f"{self.thr} nyloc nut DIN 985, A4"] += n
        self.hw[f"{self.thr} washer DIN 125, A4"] += n

    def hw_dowel(self, n=1):
        d = self.pr["dowel"]
        self.hw[f"Dowel pin {d:.0f} m6 x {20 if d == 6 else 25} ISO 8734, stainless 1.4305"] += n

    def hw_core(self, n=1, what=""):
        self.hw[f"{self.pr['core_tap']}x16 socket head ISO 4762, A4 + washer ({what})"] += n

    def section(self):
        if self._section is None:
            P = self.P
            k = P / 30
            lip, cw, ch = 2.2 * k, 16.5 * k, 6.2 * k
            sk = cq.Sketch().rect(P, P)
            for ang in (0, 90, 180, 270):
                r = math.radians(ang)
                o1 = rot2((0, P / 2 - lip / 2 + 0.1), r)
                o2 = rot2((0, P / 2 - lip - ch / 2), r)
                sk = sk.push([cq.Location(cq.Vector(o1[0], o1[1], 0), cq.Vector(0, 0, 1), ang)]).rect(
                    self.pr["slot"], lip + 0.2, mode="s").reset()
                sk = sk.push([cq.Location(cq.Vector(o2[0], o2[1], 0), cq.Vector(0, 0, 1), ang)]).rect(
                    cw, ch, mode="s").reset()
            sk = sk.push([(0, 0)]).circle(self.pr["bore"] / 2, mode="s").reset()
            self._section = sk
        return self._section

    def profile_solid(self, L):
        return cq.Workplane().placeSketch(self.section()).extrude(L).val()

    # ------------------------------------------------------------------ members
    def member(self, role, A, B, xdir, note="", colour=C_ALU):
        """Profile from A to B (centre line). Splits into equal pieces <= max_piece
        joined by a pair of dowelled splice plates."""
        A, B = V(*A), V(*B)
        L = float(np.linalg.norm(B - A))
        u = (B - A) / L
        n = max(1, math.ceil(L / self.p.max_piece - 1e-9))
        gap = 2.0
        pl = (L - (n - 1) * gap) / n
        x = nrm(np.asarray(xdir, float) - u * np.dot(xdir, u))
        y = np.cross(u, x)
        for i in range(n):
            a0 = A + u * i * (pl + gap)
            self.add(role, self.profile_solid(pl).moved(loc(a0, x, u)), colour)
            self.profiles.append(dict(role=role, length=round(pl, 1),
                                      note=note + ("; spliced end(s): through-hole 20 mm from end" if n > 1 else "")))
        for j in range(1, n):
            J = A + u * (j * pl + (j - 0.5) * gap)
            for sgn in (1, -1):
                self.plate("splice", self.splice_def(),
                           J + sgn * y * (self.P / 2) - (0 if sgn > 0 else 1) * y * self.p.plate_thk, u, y)
        return A, B, u

    # ------------------------------------------------------------------ plates
    def make_plate_local(self, outline, holes, t):
        sol = cq.Workplane().polyline(outline).close().extrude(t).val()
        for (hx, hy, d, kind) in holes:
            sol = sol.cut(cyl(d / 2, t + 2, (hx, hy, -1), (0, 0, 1)))
        return sol

    def plate(self, key, definition, origin, xdir, normal, colour=C_PLATE):
        """Place a flat plate. Plate local z = normal; plate occupies z in [0, t]."""
        if key not in self.plate_defs:
            outline, holes, material = definition
            self.plate_defs[key] = dict(outline=outline, holes=holes, material=material,
                                        solid=self.make_plate_local(outline, holes, self.p.plate_thk))
        self.plate_count[key] += 1
        for (_, _, _, kind) in self.plate_defs[key]["holes"]:
            if kind == "dowel":
                self.hw_dowel()
            elif kind == "tbolt":
                self.hw_tbolt()
            elif kind == "thru":
                self.hw_thru()
        self.add(f"plate_{key}", self.plate_defs[key]["solid"].moved(loc(origin, xdir, normal)), colour)

    def hole_row(self, direction, start, offsets, kinds):
        d, c = self.pr["dowel"], self.pr["clear"]
        return [(direction[0] * (start + o), direction[1] * (start + o), d if k == "dowel" else c, k)
                for o, k in zip(offsets, kinds)]

    def core_clear(self):
        return 8.5 if self.pr["core_tap"] == "M8" else 13.0

    def splice_def(self):
        P, h = self.P, self.p.splice_len / 2
        holes = []
        for sgn in (1, -1):
            holes += self.hole_row((sgn, 0), 1.0, (20, 55, 95), ("dowel", "tbolt", "tbolt"))
        return ([(-h, -P / 2), (h, -P / 2), (h, P / 2), (-h, P / 2)], holes, "Al 5083/6082")

    def gusset_def(self):
        """Corner node plate, drawn for the NE corner (node at origin). Same part top & bottom.
        Centre hole: leg core bolt (bottom plate). Side holes: leg-top corner brackets."""
        P, p = self.P, self.p
        w = P / 2 + 10
        members = [((-S2, -S2), p.end_gap_node, 115, 3), ((-1, 0), p.end_gap_node, 115, 3),
                   ((0, -1), p.end_gap_node, 115, 3), ((S2, S2), p.end_gap_outrigger, 80, 2)]
        pts, holes = [], []
        for (dx, dy), st, ext, nh in members:
            nx, ny = -dy, dx
            for t in (st, st + ext):
                for s in (-w, w):
                    pts.append((dx * t + nx * s, dy * t + ny * s))
            holes += self.hole_row((dx, dy), st, (20, 55, 95)[:nh], ("dowel", "tbolt", "tbolt")[:nh])
        ci = p.strut_base_inset
        for t in (ci - 27, ci + 27):
            for s in (-w, w):
                pts.append((-S2 * t + S2 * s, -S2 * t - S2 * s))
        for t in (ci - 15, ci + 15):
            holes.append((-S2 * t, -S2 * t, self.pr["clear"], "thru"))
        holes.append((0.0, 0.0, self.core_clear(), "core"))
        for (bx, by) in ((P / 2 + 14, 0.0), (0.0, P / 2 + 14)):  # leg-top brackets on the leg's outer faces
            holes.append((bx, by, self.pr["clear"], "thru"))
        pts = np.array(pts)
        hull = pts[ConvexHull(pts).vertices]
        return ([tuple(map(float, q)) for q in hull], holes, "Al 5083/6082")

    def hub_def(self, top):
        P, p = self.P, self.p
        H = 125 * P / 30
        holes = []
        for sx, sy in AP.CORNERS.values():
            dr = (sx * S2, sy * S2)
            holes += self.hole_row(dr, p.hub_gap, (45, 75, 105), ("dowel", "tbolt", "tbolt"))
            if top:
                holes.append((dr[0] * (P / 2 + 14), dr[1] * (P / 2 + 14), self.pr["clear"], "tbolt"))
        if top:
            holes.append((0.0, 0.0, self.core_clear(), "core"))
        return ([(-H, -H), (H, -H), (H, H), (-H, H)], holes, "Al 5083/6082")

    def tee_def(self):
        """Mid-side node plate. Local x = outward, y = along the side. Same part on all 4 sides."""
        P = self.P
        w = P / 2 + 10
        Ly = 1.5 + 115
        xs = P / 2 + 5 + 80
        outline = [(-w, -Ly), (w, -Ly), (w, -w), (xs, -w), (xs, w), (w, w), (w, Ly), (-w, Ly)]
        holes = []
        for sgn in (1, -1):
            holes += self.hole_row((0, sgn), 1.5, (20, 55, 95), ("dowel", "tbolt", "tbolt"))
        holes += self.hole_row((1, 0), P / 2 + 5, (20, 55), ("dowel", "tbolt"))
        return (outline, holes, "Al 5083/6082")

    def leg_base_def(self):
        """Leg foot plate: leg core bolt + 2 corner brackets + 4 levelling studs to the ground screw."""
        P = self.P
        h = 70.0
        holes = [(0.0, 0.0, self.core_clear(), "core")]
        holes += [(P / 2 + 14, 0.0, self.pr["clear"], "thru"), (-(P / 2 + 14), 0.0, self.pr["clear"], "thru")]
        holes += [(sx * 52, sy * 52, 11.0, "stud") for sx in (1, -1) for sy in (1, -1)]
        return ([(-h, -h), (h, -h), (h, h), (-h, h)], holes, "Al 5083/6082")

    # ------------------------------------------------------------------ hardware shapes
    def clevis(self, origin, xdir, normal, narrow=False):
        """Clevis bracket; local z = away from mounting face, local x in the brace plane,
        local y = pin axis. Pin at local (0, 0, pin_offset)."""
        P, po = self.P, self.p.pin_offset
        w = P / 2 + (0 if narrow else 8)
        s = box(-25, 25, -w, w, 0, 5)
        for sg in (1, -1):
            y0, y1 = sg * (P / 2 + 0.5), sg * (P / 2 + 5.5)
            s = s.fuse(box(-20, 20, min(y0, y1), max(y0, y1), 0, po + 12))
        L = loc(origin, xdir, normal)
        self.add("clevis", s.moved(L), C_STEEL)
        self.add("clevis_pin", cyl(4, P + 18, (0, -(P + 18) / 2, po), (0, 1, 0)).moved(L), C_STEEL)
        self.buy[f"Clevis bracket for {self.p.profile}, 8 mm pin (custom bent 4 mm stainless or Al, see model)"] += 1
        self.hw[f"Clevis pin 8 x {P + 20:.0f} mm DIN 1444, A4 + R-clip"] += 1

    def angle_bracket(self, origin, xdir, normal, count_bolts=True):
        """Standard corner bracket; local x along the horizontal leg, local z along vertical leg."""
        s = box(0, 28, -10, 10, 0, 4).fuse(box(0, 4, -10, 10, 0, 28))
        self.add("angle_bracket", s.moved(loc(origin, xdir, normal)), C_STEEL)
        self.buy[f"Corner bracket {self.p.profile} (28x28 cast)"] += 1
        if count_bolts:
            self.hw_tbolt(2)

    def ground_screw(self, x, y, pole=False):
        p = self.p
        top = p.screw_head_z
        s = cyl(38, 650 + top - 8, (x, y, -650), (0, 0, 1))
        if pole:
            s = s.fuse(cyl(p.pole_d / 2 + 6, 120, (x, y, top - 8), (0, 0, 1)))
            self.buy[f"Ground screw 76 x 650 mm with {p.pole_d:.1f} mm pole sleeve + clamp bolt, galvanised"] += 1
        else:
            s = s.fuse(box(x - 70, x + 70, y - 70, y + 70, top - 8, top))
            for sx in (1, -1):
                for sy in (1, -1):
                    s = s.fuse(cyl(5, p.level_gap + p.plate_thk + 12, (x + sx * 52, y + sy * 52, top), (0, 0, 1)))
            self.buy["Ground screw 76 x 650 mm with 140 x 140 mm head plate, galvanised"] += 1
            self.hw["M10 threaded rod 90 mm A4 + 3 nuts + 2 washers (levelling stud)"] += 4
        self.add("ground_screw", s, C_GROUND)

    # ------------------------------------------------------------------ mic parts
    def mic_shape(self, rear, d):
        p = self.p
        rear, d = V(*rear), nrm(d)
        body = p.mic_len - p.mic_nose_len
        s = cyl(p.mic_d / 2, body, rear, d)
        s = s.fuse(cq.Solid.makeCone(p.mic_d / 2, p.mic_tip_d / 2, p.mic_nose_len - 5,
                                     cq.Vector(*(rear + d * body)), cq.Vector(*d)))
        return s.fuse(cyl(p.mic_tip_d / 2, 5, rear + d * (p.mic_len - 5), d))

    def sleeve_shape(self, start, d):
        """TPU isolation sleeve: tube over the mic in the clamp + flange (= washer under collar)."""
        p = self.p
        start, d = V(*start), nrm(d)
        R = p.mic_d / 2 + p.iso_thk
        tube = cyl(R, p.clamp_len, start, d).cut(cyl(p.mic_d / 2, p.clamp_len + 1, start - d * 0.5, d))
        fl = cyl(R + 4, p.sleeve_flange, start + d * p.clamp_len, d).cut(
            cyl(p.mic_d / 2, p.sleeve_flange + 1, start + d * (p.clamp_len - 0.5), d))
        return tube.fuse(fl)

    def collar_shape(self):
        """Index collar: axis = +z, contact face at z=0, extends to z=10. Pinch screw M3."""
        p = self.p
        r_in, r_out = p.mic_d / 2 + 0.1, p.mic_d / 2 + 7
        s = cyl(r_out, 10, (0, 0, 0), (0, 0, 1)).fuse(box(-5, 5, r_out - 2, r_out + 7, 0, 10))
        s = s.cut(cyl(r_in, 12, (0, 0, -1), (0, 0, 1)))
        s = s.cut(box(-0.75, 0.75, 0, r_out + 8, -1, 11))
        return s.cut(cyl(1.7, 20, (-10, r_out + 3, 5), (1, 0, 0)))

    def clip_shape(self):
        """Cable strain-relief clip; frame: z away from the profile face, x along the cable."""
        p = self.p
        rc = p.cable_d / 2 + 0.4
        s = box(-12, 12, -16, 16, 0, 22)
        s = s.cut(cyl(rc, 30, (-15, 0, 15), (1, 0, 0)))
        s = s.cut(box(-13, 13, -rc + 1.2, rc - 1.2, 15, 23))
        s = s.cut(cyl(self.pr["clear"] / 2, 30, (0, 0, -1), (0, 0, 1)))
        return s.cut(cyl(6.5, 20, (0, 0, 9), (0, 0, 1)))

    def cpe_saddle_shape(self):
        """Printed saddle between the 48.3 mm pole and the CPE210's back.
        Frame: z = pointing direction (out of the pole), y = up, x = across.
        Pole axis along y at z = -r_pole; the CPE back sits on z = 0 between two side lips.
        The CPE210's own pole strap (or a 12 mm stainless hose clamp) goes round the pole,
        through the two strap slots and the CPE's strap mount, clamping all three together.
        Print on its side (x down): no overhangs in the pole cut, lips along the layers."""
        p = self.p
        rp = p.pole_d / 2
        h, w, _ = p.cpe_size
        hs = min(120.0, h * 0.55) / 2  # saddle height
        wl = w / 2 + 0.5  # lip inner face
        s = box(-wl - 4, wl + 4, -hs, hs, -rp * 0.75, 0)
        s = s.cut(cyl(rp + 0.4, 2 * hs + 2, (0, -hs - 1, -rp), (0, 1, 0)))
        for sx in (1, -1):  # side lips that key the CPE body, so it can't twist on the pole
            s = s.fuse(box(min(sx * wl, sx * (wl + 4)), max(sx * wl, sx * (wl + 4)), -hs, hs, 0, 18))
        for sy in (1, -1):  # strap slots, 14 x 3 mm, right through, just behind the CPE face
            s = s.cut(box(-wl - 6, wl + 6, sy * hs * 0.55 - 7, sy * hs * 0.55 + 7, -6, -3))
        # pointing mark: a V-notch on the top face, on the boresight line
        return s.cut(box(-1, 1, hs - 3, hs + 1, -rp * 0.75 - 1, 1))

    def holder_shape(self, base):
        """Mic holder, frame: z up, z=0 = mounting surface (outrigger top or mast top).
        base='socket': square socket over the mast end, core bolt through the floor.
        base='flange': flat flange on the profile top, 2 T-slot bolts + dowel.
        Cable exit + drain window faces local -y."""
        p, P = self.p, self.P
        sq = P + 0.6
        z_cav0 = p.holder_floor
        z_cl0 = z_cav0 + p.xlr_cavity
        z_cl1 = z_cl0 + p.clamp_len
        R = p.mic_d / 2 + p.iso_thk
        if base == "socket":
            Ds = sq * math.sqrt(2) + 8
            s = cyl(Ds / 2, p.socket_depth + p.holder_floor, (0, 0, -p.socket_depth), (0, 0, 1))
        else:
            s = box(-45, 45, -24, 24, 0, p.holder_floor)
        s = s.fuse(cyl(23, p.xlr_cavity, (0, 0, z_cav0), (0, 0, 1)))
        s = s.fuse(cq.Solid.makeCone(23, R + 5, 12, cq.Vector(0, 0, z_cl0 - 6), cq.Vector(0, 0, 1)))
        s = s.fuse(cyl(R + 5, p.clamp_len, (0, 0, z_cl0), (0, 0, 1)))
        lug_z0, lug_z1 = z_cl0 + 10, z_cl1 - 5
        s = s.fuse(box(-8, 8, R + 3, R + 13, lug_z0, lug_z1))
        if base == "socket":
            s = s.cut(box(-sq / 2, sq / 2, -sq / 2, sq / 2, -p.socket_depth - 1, 0))
            s = s.cut(cyl(self.core_clear() / 2, p.holder_floor + 2, (0, 0, -1), (0, 0, 1)))
        else:
            for xx in (34, -34):
                s = s.cut(cyl(self.pr["clear"] / 2, p.holder_floor + 2, (xx, 0, -1), (0, 0, 1)))
            s = s.cut(cyl(self.pr["dowel"] / 2, p.holder_floor + 2, (0, 0, -1), (0, 0, 1)))
        s = s.cut(cyl(18, p.xlr_cavity - 4, (0, 0, z_cav0), (0, 0, 1)))
        s = s.cut(cyl(R, p.clamp_len + 10, (0, 0, z_cl0 - 6), (0, 0, 1)))
        s = s.cut(box(-1, 1, 0, R + 14, z_cl0 + 4, z_cl1 + 1))  # clamp slit
        s = s.cut(cyl(2.2, 30, (-15, R + 8, (lug_z0 + lug_z1) / 2), (1, 0, 0)))  # M4 pinch bolt
        s = s.cut(box(-7, 7, -30, 0, z_cav0, z_cav0 + 34))  # cable exit + drain window
        return s, z_cl1

    def mic_station(self, origin, window_dir, base):
        """Holder + sleeve + collar + mic + plug + wind-screen keep-out, all pointing up.
        origin: centre of the mounting surface; window_dir: horizontal direction of cable exit."""
        p = self.p
        org = V(*origin)
        wd = nrm(window_dir)
        xdir = np.cross(wd, Z) * -1  # so that local -y = window_dir
        L = loc(org, xdir, Z)
        holder, z_cl1 = self.holder_shape(base)
        self.add(f"print_mic_holder_{base}", holder.moved(L), C_PRINT)
        self.printed[f"mic_holder_{base}"] += 1
        clamp_top = org + Z * z_cl1
        tip = clamp_top + Z * (p.sleeve_flange + p.collar_from_tip)
        rear = tip - Z * p.mic_len
        self.add("mic", self.mic_shape(rear, Z), C_MIC)
        self.add("print_sleeve_tpu", self.sleeve_shape(clamp_top - Z * p.clamp_len, Z), C_TPU)
        self.printed["iso_sleeve_tpu"] += 1
        self.add("print_index_collar", self.collar_shape().moved(
            cq.Location(cq.Vector(*(tip - Z * p.collar_from_tip)))), C_PRINT)
        self.printed["index_collar"] += 1
        self.hw["M3x16 socket head + M3 nut, A4 (collar)"] += 1
        self.hw["M4x30 socket head + M4 nyloc, A4 (holder clamp)"] += 1
        self.add("xlr_plug", cyl(10, p.xlr_plug_len, rear - Z * p.xlr_plug_len, Z), C_MIC)
        ac = tip - Z * p.ac_from_tip
        self.add("windscreen_keepout", cq.Solid.makeSphere(p.windscreen_d / 2, cq.Vector(*ac)), C_WIND)
        if base == "socket":
            self.hw_core(1, "apex holder -> mast")
        else:
            self.hw_tbolt(2)
            self.hw_dowel(1)
        return ac

    # ------------------------------------------------------------------ build
    def build(self):
        p, g, P = self.p, self.g, self.P
        a, zc, t = g.a, g.z0 + P / 2, p.plate_thk

        # ---- hub ----
        self.plate("hub_top", self.hub_def(True), V(0, 0, g.z1), V(1, 0, 0), Z)
        self.plate("hub_bottom", self.hub_def(False), V(0, 0, g.z0 - t), V(1, 0, 0), Z)

        corner_ang = {"NE": 0, "SE": -90, "SW": 180, "NW": 90}
        out_inner = p.end_gap_outrigger
        acs = []
        for idx, (name, (sx, sy)) in enumerate(AP.CORNERS.items()):
            d = V(sx * S2, sy * S2, 0)
            C = V(sx * a, sy * a, zc)
            perp = np.cross(Z, d)
            self.member("arm", d * p.hub_gap + V(0, 0, zc), C - d * p.end_gap_node, perp,
                        note="H7 through-hole 20 mm from outer end, 45 mm from hub end")
            self.member("outrigger", C + d * out_inner, C + d * (p.outrigger + p.outrigger_tail), perp,
                        note=f"H7 through-holes 20 mm from inner end and {p.outrigger - out_inner:.0f} mm "
                             f"(mic holder dowel)")
            sides = (V(-sx, 0, 0), V(0, -sy, 0))
            for e in sides:
                M = C + e * a
                self.member("side_half", C + e * p.end_gap_node, M - e * 1.5, np.cross(Z, e),
                            note=f"H7 through-holes 20 mm from each end and {p.knee - p.end_gap_node:.0f} mm "
                                 f"from the corner end (knee-brace clevis, underside)")
            ang = math.radians(corner_ang[name])
            gx = V(math.cos(ang), math.sin(ang), 0)
            self.plate("corner_gusset", self.gusset_def(), V(C[0], C[1], g.z1), gx, Z)
            self.plate("corner_gusset", self.gusset_def(), V(C[0], C[1], g.z0 - t), gx, Z)

            # ---- leg, knee braces, ground screw ----
            self.member("leg", V(C[0], C[1], g.leg_bottom), V(C[0], C[1], g.leg_top), V(1, 0, 0),
                        note=f"tap core {self.pr['core_tap']} both ends; H7 through-holes at "
                             f"{g.leg_top - p.knee - g.leg_bottom:.0f} mm from the foot (knee clevises, 2 faces)")
            self.hw_core(2, "leg -> gusset / base plate")
            for e in (V(sx, 0, 0), V(0, sy, 0)):  # leg-top brackets on the outer faces (holes in gusset)
                self.angle_bracket(V(C[0], C[1], g.leg_top) + e * (P / 2), e, -Z, count_bolts=False)
                self.hw_tbolt(1)
            self.plate("leg_base", self.leg_base_def(), V(C[0], C[1], g.leg_bottom - t), V(1, 0, 0), Z)
            for e in (V(sx, 0, 0), V(0, sy, 0)):
                self.angle_bracket(V(C[0], C[1], g.leg_bottom) + e * (P / 2), e, Z, count_bolts=False)
                self.hw_tbolt(1)
            self.ground_screw(C[0], C[1])
            for e in sides:
                zk = g.leg_top - p.knee
                o_leg = V(C[0], C[1], zk) + e * (P / 2)
                self.clevis(o_leg, Z, e, narrow=True)
                self.hw_dowel(1)
                self.hw_tbolt(2)
                o_side = V(C[0], C[1], g.z0) + e * p.knee
                self.clevis(o_side, e, -Z, narrow=True)
                self.hw_dowel(1)
                self.hw_tbolt(2)
                A = o_leg + e * p.pin_offset
                B = o_side - Z * p.pin_offset
                u = nrm(B - A)
                self.member("knee_brace", A - u * 15, B + u * 15, np.cross(Z, e),
                            note=f"8 mm H8 cross-holes 15 mm from each end (pin-to-pin {np.linalg.norm(B - A):.1f} mm)")

            # ---- strut ----
            stag = 0.0 if idx % 2 == 0 else 50.0
            base_o = V(C[0], C[1], g.zg) - d * p.strut_base_inset
            self.clevis(base_o, d, Z)
            Apin = base_o + Z * p.pin_offset
            top_o = d * (P / 2) + V(0, 0, g.z_strut_top - stag)
            self.clevis(top_o, Z, d, narrow=True)
            self.hw_dowel(1)
            self.hw_tbolt(2)
            Bpin = top_o + d * p.pin_offset
            u = nrm(Bpin - Apin)
            self.member(f"strut_{'A' if stag == 0 else 'B'}", Apin - u * 15, Bpin + u * 15, perp,
                        note="8 mm H8 cross-holes for clevis pins 15 mm from each end (pin-to-pin "
                             f"{np.linalg.norm(Bpin - Apin):.1f} mm)")

            # ---- base mic on the outrigger end ----
            mic_org = V(C[0], C[1], g.z1) + d * p.outrigger
            acs.append(self.mic_station(mic_org, -d, "flange"))
            clip_org = V(C[0], C[1], g.z1) + d * (p.outrigger - 150)
            self.add("print_cable_clip", self.clip_shape().moved(loc(clip_org, d, Z)), C_PRINT)
            self.printed["cable_clip"] += 1
            self.hw_tbolt(1, length=25)

        # ---- mid-side nodes ----
        for sname, (sx, sy) in AP.SIDES.items():
            n = V(sx, sy, 0)
            M = n * a
            self.plate("midside_tee", self.tee_def(), V(M[0], M[1], g.z1), n, Z)
            self.plate("midside_tee", self.tee_def(), V(M[0], M[1], g.z0 - t), n, Z)

        # ---- mast + apex ----
        xm = nrm(V(1, 1, 0))
        self.member("mast", V(0, 0, g.mast_base), V(0, 0, g.mast_top), xm,
                    note=f"tap core {self.pr['core_tap']} both ends; H7 dowel holes at clevis heights "
                         f"{g.z_strut_top - g.mast_base:.0f} and {g.z_strut_top - 50 - g.mast_base:.0f} mm from foot")
        self.hw_core(1, "mast -> hub")
        for sx, sy in AP.CORNERS.values():
            d = V(sx * S2, sy * S2, 0)
            self.angle_bracket(d * (P / 2) + V(0, 0, g.zg), d, Z, count_bolts=False)
            self.hw_tbolt(1)  # the other bolt is a hub-plate hole
        acs.append(self.mic_station(V(0, 0, g.mast_top), V(1, -1, 0), "socket"))
        dface = nrm(V(1, -1, 0))
        corg = dface * (P / 2) + V(0, 0, g.mast_top - p.socket_depth - 45)
        self.add("print_cable_clip", self.clip_shape().moved(loc(corg, Z, dface)), C_PRINT)
        self.printed["cable_clip"] += 1
        self.hw_tbolt(1, length=25)

        # ---- mic 6 (optional) ----
        if p.mic6:
            n = V(*AP.SIDES[p.mic6_side], 0)
            M = n * a + V(0, 0, zc)
            st = M + n * (P / 2 + 5)
            self.member("outrigger", st, st + n * (p.outrigger + p.outrigger_tail - out_inner),
                        np.cross(Z, n), note="(mic 6) same as corner outriggers")
            acs.append(self.mic_station(V(*(n * (a + p.outrigger)))[:2].tolist() + [g.z1], -n, "flange"))
        self.checks["mic_positions_model_vs_params"] = float(np.abs(
            np.array(acs) - np.array([m["pos"] for m in g.mics])).max())

        # ---- electronics / battery box on a leg ----
        cx, cy = AP.CORNERS[p.box_corner]
        C = V(cx * a, cy * a, 0)
        out = V(cx, 0, 0)  # outward-facing (west for SW) leg face: no brace, no panel there
        bw, bh, bd = p.box_size
        org = C + out * (P / 2) + Z * p.box_z
        pl = box(-bh / 2 - 20, bh / 2 + 20, -bw / 2 - 20, bw / 2 + 20, 0, 4)  # local x = up, y = along y
        self.add("box_plate", pl.moved(loc(org, Z, out)), C_PLATE)
        self.add("electronics_box", box(-bh / 2, bh / 2, -bw / 2, bw / 2, 4, 4 + bd).moved(loc(org, Z, out)), C_BOX)
        self.buy[f"Al plate {bh + 40:.0f} x {bw + 40:.0f} x 4 mm (box mounting plate on leg)"] += 1
        self.buy[f"Weatherproof enclosure {bw:.0f}x{bh:.0f}x{bd:.0f} mm IP66, vent membrane + cable glands"] += 1
        self.hw_tbolt(4)
        self.hw["M5x16 + washer A4 (box to plate)"] += 4
        self.box_centre = org + out * (4 + bd / 2)

        # ---- solar panels on the south side ----
        if p.solar:
            self.solar_stand()

        # ---- pole: wind sensor + Wi-Fi bridge + camera ----
        if p.pole_side:
            n = V(*AP.SIDES[p.pole_side], 0)
            tng = np.cross(Z, n)
            M = n * a + V(0, 0, zc)
            st0 = M + n * (P / 2 + 5)
            self.member("pole_stub", st0, st0 + n * 250, tng, note="H7 through-hole 20 mm from inner end")
            post_c = st0 + n * 200
            self.member("pole_post", V(post_c[0], post_c[1], g.z1), V(post_c[0], post_c[1], g.z1 + p.post_len), n,
                        note="tap core bottom end")
            self.hw_core(1, "post -> stub")
            for s2 in (1, -1):
                self.angle_bracket(V(post_c[0], post_c[1], g.z1) + tng * s2 * (P / 2), tng * s2, Z)
            pc = post_c + n * (P / 2 + p.pole_d / 2 + 6)
            pc[2] = 0.0
            self.ground_screw(pc[0], pc[1], pole=True)
            self.add("pole", cyl(p.pole_d / 2, p.pole_len - p.screw_head_z, (pc[0], pc[1], p.screw_head_z),
                                 (0, 0, 1)), C_POLE)
            self.buy[f"Aluminium or galvanised tube {p.pole_d:.1f} mm, {p.pole_len / 1000 + 0.2:.1f} m (2 parts + "
                     f"internal sleeve joint for transport)"] += 1
            for zz in (g.z1 + 120, g.z1 + p.post_len - 60):
                ring = cyl(p.pole_d / 2 + 4, 20, (pc[0], pc[1], zz), (0, 0, 1)).cut(
                    cyl(p.pole_d / 2, 22, (pc[0], pc[1], zz - 1), (0, 0, 1)))
                blk = box(-12, 12, -12, 12, 0, 20).moved(cq.Location(cq.Vector(*(post_c + n * (P / 2 + 3))[:2], zz)))
                self.add("pole_clamp", ring.fuse(blk), C_STEEL)
            self.buy[f"U-bolt pole clamp for {p.pole_d:.1f} mm pole onto {p.profile} slot, A4"] += 2
            self.hw_tbolt(4)
            sd, sh = p.wind_sensor
            self.add("wind_sensor", cyl(sd / 2, sh, (pc[0], pc[1], p.pole_len), (0, 0, 1)).fuse(
                cyl(p.pole_d / 2, 40, (pc[0], pc[1], p.pole_len - 40), (0, 0, 1))), C_SENSOR)
            self.buy[f"Ultrasonic 2D wind sensor (e.g. Gill WindSonic class), ~{sd:.0f} mm, pole-top mount"] += 1
            # Wi-Fi bridge dish and sky camera placeholders, below the sensor
            zc_ = p.pole_len - 600
            # TP-Link CPE210 pointing the same way as the solar panels (true south by default)
            az = math.radians(p.cpe_azimuth)
            f = V(math.sin(az), math.cos(az), 0)  # boresight, horizontal
            xd = np.cross(Z, f)
            ch, cw, cdp = p.cpe_size
            o = pc + f * (p.pole_d / 2) + Z * zc_  # saddle front face (CPE back) on the boresight line
            L = loc(o, xd, f)
            self.add("cpe_pole_saddle", self.cpe_saddle_shape().moved(L), C_PRINT)
            self.printed["cpe_pole_saddle"] += 1
            self.add("wifi_bridge_cpe210", box(-cw / 2, cw / 2, -ch / 2, ch / 2, 0, cdp).moved(L), C_SENSOR)
            self.buy["TP-Link CPE210 (2.4 GHz, 9 dBi, 65 deg H / 35 deg V beam) + its pole strap"] += 1
            self.buy["Hose clamp 12 mm W4 stainless, 40-70 mm (spare strap for the CPE saddle)"] += 1
            self.checks["cpe_boresight_azimuth_deg"] = float(math.degrees(math.atan2(f[0], f[1])) % 360)
            self.add("sky_camera", cyl(35, 90, pc - n * (p.pole_d / 2 + 50) + Z * (zc_ + 250), Z), C_SENSOR)
            self.buy["Sky camera + pole bracket (user supplied)"] += 1
            self.pole_xy = pc[:2]
            self.sensor_z = p.pole_len + sh / 2

        self.buy["Armoured / steel-braided cable conduit, marten-proof (all cables, legs to box)"] += 1

    def solar_stand(self):
        """Separate ground stand south of the array (not on the array frame: ~2 m2 of panel is a
        sail that would rock the frame and shake the mics, and a big reflector next to them).
        Two A-frames (front post, back post, inclined rail, tie + diagonal), each post on a
        ground screw; two cross rails carry the panels, portrait, facing true south (-y)."""
        p, g, P = self.p, self.g, self.P
        t = p.plate_thk
        beta = math.radians(p.panel_tilt)
        sb, cb = math.sin(beta), math.cos(beta)
        pw, plen, pth = p.panel_size
        nP = p.panel_count
        d_r = V(0, -cb, -sb)  # down the slope (north -> south)
        n_p = V(0, -sb, cb)  # panel normal (front faces south + up)
        Lr = p.panel_rail
        W = nP * pw + (nP - 1) * 20
        xr = W / 2 - 250
        y_n = -g.a - P / 2 - p.solar_gap  # back (north) posts
        off = 1.5 * P + pth / 2  # rail centreline -> panel mid-plane
        z_T = p.panel_low + sb * (Lr / 2 + plen / 2) - cb * off  # panel bottom edge at panel_low
        T = V(0, y_n, z_T)
        B = T + d_r * Lr
        z_tie = g.leg_bottom + 280
        for sx in (1, -1):
            x = sx * xr
            Tx, Bx = V(x, T[1], T[2]), V(x, B[1], B[2])
            self.member("solar_rail", Tx, Bx, V(1, 0, 0), note="H7 holes at both ends (post brackets)")
            for top in (Tx, Bx):
                foot = V(x, top[1], g.leg_bottom)
                self.member("solar_post", foot, top - V(0, 0, P / 2), V(1, 0, 0),
                            note=f"tap core {self.pr['core_tap']} at the foot")
                self.hw_core(1, "solar post -> base plate")
                self.plate("leg_base", self.leg_base_def(), foot - V(0, 0, t), V(1, 0, 0), Z)
                self.angle_bracket(foot + V(sx * P / 2, 0, 0), V(sx, 0, 0), Z)
                self.angle_bracket(top - V(0, 0, P / 2) + V(sx * P / 2, 0, 0), V(sx, 0, 0), -Z)
                self.ground_screw(x, top[1])
            xs = x + sx * P  # tie + diagonal on the outer face of the posts
            self.member("solar_tie", V(xs, T[1], z_tie), V(xs, B[1], z_tie), V(0, 0, 1),
                        note="bolt through the post side slots (corner brackets)")
            self.member("solar_diagonal", V(xs, T[1], z_tie), V(xs, B[1], B[2] - P), V(0, 0, 1),
                        note="drilled ends, M6 through-bolt to the post")
            self.hw["M6x50 + nyloc A4 (solar diagonal)"] += 2
        # east-west bracing between the back posts
        self.member("solar_back_tie", V(-xr, y_n + P, z_tie), V(xr, y_n + P, z_tie), V(0, 1, 0),
                    note="corner brackets to the back posts")
        self.member("solar_back_diagonal", V(-xr, y_n + P, z_tie + P), V(xr, y_n + P, z_T - 2 * P), V(0, 1, 0),
                    note="drilled ends, M6 through-bolt to the posts")
        mid = T + d_r * (Lr / 2)
        for s_ in (-plen / 4, plen / 4):
            c = mid + d_r * s_ + n_p * P
            self.member("solar_cross_rail", c - V(W / 2, 0, 0), c + V(W / 2, 0, 0), n_p,
                        note="panel clamps; bolt to the inclined rails with corner brackets")
            for sx in (1, -1):
                self.angle_bracket(V(sx * xr, c[1], c[2]) - n_p * (P / 2), V(-sx, 0, 0), n_p)
        self.hw["PV mid/end clamp 35 mm frame, aluminium + M8 T-bolt"] += 2 * (nP + 1)
        panel_c0 = mid + n_p * off
        for i in range(nP):
            pc = panel_c0 + V((i - (nP - 1) / 2) * (pw + 20), 0, 0)
            frame = box(-pw / 2, pw / 2, -plen / 2, plen / 2, -pth / 2, pth / 2)
            self.add("solar_panel", frame.moved(loc(pc, V(1, 0, 0), n_p)), C_PV)
        self.buy[f"PV panel ~200 W 12 V mono, {pw:.0f} x {plen:.0f} x {pth:.0f} mm"] += nP
        self.buy["MPPT charge controller 12 V 20 A with low-temperature charge cut-off (LiFePO4)"] += 1
        self.buy["LiFePO4 battery 12.8 V 100 Ah (~1.3 kWh, ~2 days at the ~25 W system load), in the box"] += 1
        self.buy[f"PV cable 4 mm2 ~{(p.solar_gap + 3000) / 1000:.0f} m in buried/armoured conduit + MC4 + 15 A fuse"] += 1
        top_edge = panel_c0 - d_r * (plen / 2) + n_p * (pth / 2)
        bot_edge = panel_c0 + d_r * (plen / 2)
        self.checks["panel_top_edge_z"] = float(top_edge[2])
        self.checks["panel_bottom_edge_z"] = float(bot_edge[2])
        mic_y = min(m["pos"][1] for m in g.mics)
        self.checks["solar_stand_to_nearest_mic_y"] = float(mic_y - top_edge[1])

    # ------------------------------------------------------------------ outputs
    def export_step(self, path):
        assy = cq.Assembly(name="acoustic_array")
        for name, shape, col in self.parts:
            assy.add(shape, name=name, color=cq.Color(*col))
        assy.export(path)

    def export_printed(self, d):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".stl"):
                os.remove(os.path.join(d, f))
        flange, _ = self.holder_shape("flange")
        socket, _ = self.holder_shape("socket")
        flip = cq.Location(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180)
        out = {
            "mic_holder_flange": flange,  # prints flange-down
            "mic_holder_socket": socket.moved(flip),  # prints socket-up
            "iso_sleeve_tpu": self.sleeve_shape(V(0, 0, 0), (0, 0, 1)),
            "index_collar": self.collar_shape(),
            "cable_clip": self.clip_shape(),
            "cpe_pole_saddle": self.cpe_saddle_shape(),
        }
        for k, s in out.items():
            cq.exporters.export(cq.Workplane().add(s), os.path.join(d, f"{k}.stl"), tolerance=0.02, angularTolerance=0.1)
        return list(out)

    def export_dxf(self, d):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".dxf"):
                os.remove(os.path.join(d, f))
        for key, pd in self.plate_defs.items():
            sec = cq.Workplane().add(pd["solid"]).section(self.p.plate_thk / 2)
            cq.exporters.export(sec, os.path.join(d, f"{key}.dxf"))

    def export_bom(self, path):
        p = self.p
        rows = []
        grp = collections.OrderedDict()
        for pr in self.profiles:
            k = (pr["role"], pr["length"], pr["note"])
            grp[k] = grp.get(k, 0) + 1
        total = 0.0
        for (role, L, note), q in grp.items():
            rows.append(("profile", f"{p.profile} extrusion — {role}", f"cut {L:.1f} mm, square ends", q, note))
            total += L * q
        rows.append(("profile", f"{p.profile} extrusion — total length", f"{total / 1000:.2f} m (+ saw kerf)", "", ""))
        for key, q in self.plate_count.items():
            pdef = self.plate_defs[key]
            pts = np.array(pdef["outline"])
            ext = pts.max(0) - pts.min(0)
            nd = sum(1 for h in pdef["holes"] if h[3] == "dowel")
            rows.append(("plate", key, f"{pdef['material']}, t={p.plate_thk:.0f} mm, envelope {ext[0]:.0f} x "
                                       f"{ext[1]:.0f} mm, DXF plates/{key}.dxf", q,
                         f"{nd} dowel holes {self.pr['dowel']:.0f} H7 (ream), other holes per DXF"))
        for k, q in self.buy.items():
            rows.append(("bought", k, "", q, ""))
        for k, q in sorted(self.hw.items()):
            if q:
                rows.append(("fastener", k, "", q, "add ~10 % spares"))
        mats = {"mic_holder_flange": "ASA, 5 walls, 50 % infill, print flange-down",
                "mic_holder_socket": "ASA, 5 walls, 50 % infill, print socket-up",
                "iso_sleeve_tpu": "TPU 85A-95A (or cut from EPDM tube 22x28 mm)",
                "index_collar": "ASA or PETG", "cable_clip": "ASA/PETG (TPU also works)",
                "cpe_pole_saddle": "ASA, 4 walls, 40 % infill, print on its side"}
        for k, q in self.printed.items():
            rows.append(("printed", k, mats.get(k, ""), q, f"stl/{k}.stl"))
        nm = 5 + int(p.mic6)
        rows.append(("bought", "Foam windscreen + furry cover", f"~{p.windscreen_d:.0f} mm ball", nm, ""))
        rows.append(("bought", "XLR cable, outdoor PUR", "length to suit box", nm, ""))
        with open(path, "w") as f:
            f.write("category,item,spec,qty,notes\n")
            for r in rows:
                f.write(",".join('"' + str(c).replace('"', "'") + '"' for c in r) + "\n")
        return rows

    def export_render_meshes(self, d):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        groups = collections.defaultdict(list)
        for name, shape, col in self.parts:
            groups[col].append(shape)
        files = []
        for i, (col, shapes) in enumerate(groups.items()):
            fn = os.path.join(d, f"group_{i}.stl")
            cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound(shapes)), fn,
                                tolerance=0.3, angularTolerance=0.3)
            files.append((fn, col))
        return files


def render(files, outdir, g, p, pole=None):
    exe = shutil.which("openscad")
    if not exe:
        print("openscad not found; skipping renders")
        return []
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.endswith(".png"):
            os.remove(os.path.join(outdir, f))
    scad = os.path.join(outdir, "assembly.scad")
    with open(scad, "w") as f:
        f.write("color([0.55,0.65,0.45,1]) translate([-3500,-6000,-2]) cube([7000,9000,2]);\n")  # lawn
        for fn, col in files:
            f.write(f'color([{col[0]},{col[1]},{col[2]},{col[3]}]) import("{fn}");\n')
    m0 = g.mics[1]["pos"]  # SE base mic
    hz = g.z_apex / 2
    views = {
        "iso_southeast": (f"8200,-11000,{hz + 4200:.0f},0,-1600,{hz - 300:.0f}", "1600,1400", "p"),
        "iso_northwest": (f"-7600,7600,{hz + 3000:.0f},0,-1200,{hz - 200:.0f}", "1600,1400", "p"),
        "south_elevation": (f"0,-16000,{hz + 300:.0f},0,0,{hz + 300:.0f}", "1400,1500", "o"),
        "west_elevation": (f"-16000,-1600,{hz + 300:.0f},0,-1600,{hz + 300:.0f}", "1600,1300", "o"),
        "top": (f"0,-1600,16000,0,-1600,0", "1200,1500", "o"),
        "base_mic_closeup": (f"{m0[0] + 700:.0f},{m0[1] - 900:.0f},{m0[2] + 450:.0f},{m0[0] - 80:.0f},"
                             f"{m0[1] + 80:.0f},{m0[2] - 120:.0f}", "1200,1200", "p"),
        "apex_closeup": (f"900,-1200,{g.z_apex + 250:.0f},0,0,{g.mast_top + 60:.0f}", "1000,1300", "p"),
    }
    out = []
    if pole is not None:
        views["cpe_closeup"] = (f"{pole[0] + 900:.0f},{pole[1] - 1300:.0f},{p.pole_len - 200:.0f},{pole[0]:.0f},{pole[1] - 60:.0f},{p.pole_len - 600:.0f}", "1000,1000", "p")
    for name, (cam, size, proj) in views.items():
        png = os.path.join(outdir, f"{name}.png")
        cmd = [exe, "-o", png, f"--imgsize={size}", f"--camera={cam}", "--colorscheme=Tomorrow",
               f"--projection={proj}"] + [scad]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-800:])
        out.append(png)
    return out


def parse_overrides(items):
    p = AP.DEFAULT
    kw = {}
    names = {f.name for f in fields(AP.Params)}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown parameter {k}")
        cur = getattr(p, k)
        if isinstance(cur, bool):
            kw[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            kw[k] = int(v)
        elif isinstance(cur, float):
            kw[k] = float(v)
        elif isinstance(cur, tuple):
            kw[k] = tuple(float(x) for x in v.split(","))
        else:
            kw[k] = v
    return replace(p, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help="override a Params field, e.g. --set apex_rise=1600")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    p = parse_overrides(args.set)
    os.makedirs(OUT, exist_ok=True)

    m = Model(p)
    m.build()
    g = m.g

    AP.write_mic_files(p, os.path.join(OUT, "mic_positions.json"), os.path.join(OUT, "mic_positions.csv"))
    m.export_step(os.path.join(OUT, "array_assembly.step"))
    printed = m.export_printed(os.path.join(OUT, "stl"))
    m.export_dxf(os.path.join(OUT, "plates"))
    m.export_bom(os.path.join(OUT, "bom.csv"))

    mics = np.array([mm["pos"] for mm in g.mics])
    print(f"parts: {len(m.parts)}  printed STLs: {printed}")
    print(f"longest profile piece: {max(pr['length'] for pr in m.profiles):.0f} mm (limit {p.max_piece:.0f})")
    print(f"electronics box centre -> nearest mic: {np.linalg.norm(mics - m.box_centre, axis=1).min():.0f} mm")
    if hasattr(m, "pole_xy"):
        print(f"pole -> nearest mic (horizontal): {np.linalg.norm(mics[:, :2] - m.pole_xy, axis=1).min():.0f} mm; "
              f"wind sensor centre {m.sensor_z:.0f} mm, apex capsule {g.z_apex:.0f} mm")
    for k, v in m.checks.items():
        print(f"check {k}: {v:.1f}")
    if not args.no_render:
        files = m.export_render_meshes(os.path.join(OUT, "render", "mesh"))
        pngs = render(files, os.path.join(OUT, "render"), g, p, getattr(m, "pole_xy", None))
        print("renders:", [os.path.basename(x) for x in pngs])


if __name__ == "__main__":
    main()
