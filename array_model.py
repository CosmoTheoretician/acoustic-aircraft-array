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
- Base = 4 diagonal "arms" from a central hub to 4 corner nodes + a perimeter square
  (each side in two halves joined at a mid-side node). Ground mics sit on "outriggers"
  that continue the diagonals 450 mm beyond the corner nodes, so no frame member or
  strut foot is near a capsule.
- Mast stands on the hub; 4 struts run from the corner nodes to the mast top (pyramid
  edges). Struts end in clevis pins: the triangle hub-node-mast is closed by fixed
  lengths, so pivots don't add play, and pins assemble without tools.
- Repeatability: every profile has a jig-drilled H7 through-hole 20 mm from each joint
  end; plates carry matching reamed holes and dowel pins. Bolts + T-nuts only clamp.
  Strut length is set by drilled pin holes. Nothing relies on a slotted position.
- Mics are held in rubber/TPU sleeves inside printed clamps; an index collar fixed on
  each mic sets its axial position. Ground mics lie horizontally with the axis 16 mm
  above a hard board (boundary / ground-plane mounting), tip pointing outward.
"""

import argparse
import collections
import math
import os
import shutil
import subprocess
from dataclasses import asdict, fields, replace

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
C_BOARD = (0.92, 0.90, 0.84, 1)
C_WIND = (0.55, 0.55, 0.60, 0.25)  # wind-screen keep-out (not a part)
C_FOOT = (0.10, 0.10, 0.10, 1)
C_BOX = (0.80, 0.82, 0.78, 1)
C_POLE = (0.35, 0.36, 0.38, 1)

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
        self.hw = collections.Counter()  # hardware -> qty
        self.printed = collections.Counter()
        self.buy = collections.Counter()
        self.feet = []
        self._section = None
        self.names = collections.Counter()

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
                self.plate("splice", self.splice_def(), J + sgn * y * (self.P / 2) - (0 if sgn > 0 else 1) * y * self.p.plate_thk,
                           u, y)
        return A, B, u

    # ------------------------------------------------------------------ plates
    def make_plate_local(self, outline, holes, t):
        wp = cq.Workplane().polyline(outline).close().extrude(t)
        sol = wp.val()
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
        d = self.pr["dowel"]
        c = self.pr["clear"]
        out = []
        for o, k in zip(offsets, kinds):
            px, py = direction[0] * (start + o), direction[1] * (start + o)
            out.append((px, py, d if k == "dowel" else c, k))
        return out

    def splice_def(self):
        P, h = self.P, self.p.splice_len / 2
        holes = []
        for sgn in (1, -1):
            holes += self.hole_row((sgn, 0), 1.0, (20, 55, 95), ("dowel", "tbolt", "tbolt"))
        return ([(-h, -P / 2), (h, -P / 2), (h, P / 2), (-h, P / 2)], holes, "Al 5083/6082")

    def gusset_def(self):
        """Corner node plate, drawn for the NE corner (node at origin). Same part top & bottom."""
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
        # strut clevis base, centred strut_base_inset inward on the diagonal
        ci = p.strut_base_inset
        for t in (ci - 27, ci + 27):
            for s in (-w, w):
                pts.append((-S2 * t + S2 * s, -S2 * t - S2 * s))
        for t in (ci - 15, ci + 15):
            holes.append((-S2 * t, -S2 * t, self.pr["clear"], "thru"))
        holes.append((0.0, 0.0, 6.8 if self.pr["foot"] == "M8" else 8.5, "foot"))
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
            if top:  # mast foot bracket bolt into the arm's T-nut
                holes.append((dr[0] * (P / 2 + 14), dr[1] * (P / 2 + 14), self.pr["clear"], "tbolt"))
        if top:
            holes.append((0.0, 0.0, 8.5 if self.pr["core_tap"] == "M8" else 13.0, "core"))
        else:
            holes.append((0.0, 0.0, 6.8 if self.pr["foot"] == "M8" else 8.5, "foot"))
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
        holes.append((0.0, 0.0, 6.8 if self.pr["foot"] == "M8" else 8.5, "foot"))
        return (outline, holes, "Al 5083/6082")

    # ------------------------------------------------------------------ hardware shapes
    def foot(self, x, y, z_top):
        r = 4 if self.pr["foot"] == "M8" else 5
        s = cyl(25, 8, (x, y, 0), (0, 0, 1)).fuse(cyl(r, z_top - 8, (x, y, 8), (0, 0, 1)))
        self.add("foot", s, C_FOOT)
        self.feet.append((x, y))

    def clevis(self, origin, xdir, normal, mast=False):
        """Clevis bracket; local z = away from mounting face, local x in the strut plane,
        local y = pin axis. Pin at local (0, 0, pin_offset)."""
        P, po = self.P, self.p.pin_offset
        w = P / 2 + (0 if mast else 8)
        s = box(-25, 25, -w, w, 0, 5)
        for sg in (1, -1):
            y0 = sg * (P / 2 + 0.5)
            y1 = sg * (P / 2 + 5.5)
            s = s.fuse(box(-20, 20, min(y0, y1), max(y0, y1), 0, po + 12))
        self.add("clevis", s.moved(loc(origin, xdir, normal)), C_STEEL)
        pin = cyl(4, P + 18, (0, -(P + 18) / 2, po), (0, 1, 0))
        self.add("clevis_pin", pin.moved(loc(origin, xdir, normal)), C_STEEL)
        self.buy[f"Clevis bracket for {self.p.profile} strut, pin {8} mm (custom bent 4 mm stainless or Al, see model)"] += 1
        self.hw[f"Clevis pin 8 x {P + 20:.0f} mm DIN 1444, A4 + R-clip"] += 1

    def angle_bracket(self, origin, xdir, normal):
        """Standard corner bracket; local x along the horizontal leg, local z along vertical leg."""
        s = box(0, 28, -10, 10, 0, 4).fuse(box(0, 4, -10, 10, 0, 28))
        self.add("angle_bracket", s.moved(loc(origin, xdir, normal)), C_STEEL)
        self.buy[f"Corner bracket {self.p.profile} (28x28 cast), with 2 bolts + T-nuts"] += 1
        self.hw_tbolt(2)

    # ------------------------------------------------------------------ mic parts
    def mic_shape(self, rear, d):
        p = self.p
        rear, d = V(*rear), nrm(d)
        body = p.mic_len - p.mic_nose_len
        s = cyl(p.mic_d / 2, body, rear, d)
        s = s.fuse(cq.Solid.makeCone(p.mic_d / 2, p.mic_tip_d / 2, p.mic_nose_len - 5,
                                     cq.Vector(*(rear + d * body)), cq.Vector(*d)))
        s = s.fuse(cyl(p.mic_tip_d / 2, 5, rear + d * (p.mic_len - 5), d))
        return s

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
        """Index collar in its own frame: axis = +z, contact face at z=0, extends to z=10."""
        p = self.p
        r_in, r_out = p.mic_d / 2 + 0.1, p.mic_d / 2 + 7
        s = cyl(r_out, 10, (0, 0, 0), (0, 0, 1)).fuse(box(-5, 5, r_out - 2, r_out + 7, 0, 10))
        s = s.cut(cyl(r_in, 12, (0, 0, -1), (0, 0, 1)))
        s = s.cut(box(-0.75, 0.75, 0, r_out + 8, -1, 11))
        s = s.cut(cyl(1.7, 20, (-10, r_out + 3, 5), (1, 0, 0)))
        return s

    def cradle_shapes(self):
        """Ground-mic clamp in the mic frame (x = mic axis outward, z=0 board top)."""
        p, g = self.p, self.g
        R = p.mic_d / 2 + p.iso_thk
        x1 = p.ac_from_tip - p.collar_from_tip - p.sleeve_flange
        x0 = x1 - p.clamp_len
        h, W, sp = g.axis_h, 33.0, 0.5
        base = box(x0, x1, -W, W, 0, h - sp)
        cap = box(x0, x1, -W, W, h + sp, h + R + 5)
        try:
            cap = cq.Workplane().add(cap).edges("|X and >Z").fillet(8).val()
        except Exception:
            pass
        bore = cyl(R, p.clamp_len + 2, (x0 - 1, 0, h), (1, 0, 0))
        base, cap = base.cut(bore), cap.cut(bore)
        for y in (25, -25):
            hole = cyl(2.75, 100, ((x0 + x1) / 2, y, -10), (0, 0, 1))
            base, cap = base.cut(hole), cap.cut(hole)
        return base, cap, (x0, x1)

    def clip_shape(self):
        """Cable strain-relief clip, frame: z away from the profile face, x along cable."""
        p = self.p
        rc = p.cable_d / 2 + 0.4
        s = box(-12, 12, -16, 16, 0, 22)
        s = s.cut(cyl(rc, 30, (-15, 0, 15), (1, 0, 0)))
        s = s.cut(box(-13, 13, -rc + 1.2, rc - 1.2, 15, 23))
        s = s.cut(cyl(self.pr["clear"] / 2, 30, (0, 0, -1), (0, 0, 1)))
        s = s.cut(cyl(6.5, 20, (0, 0, 9), (0, 0, 1)))
        return s

    def apex_holder_shape(self):
        """Apex mic holder in the mast-top frame (z up, mast top at z=0, square socket on local axes)."""
        p, P = self.p, self.P
        sq = P + 0.6
        Ds = sq * math.sqrt(2) + 8
        z_cav0 = p.holder_floor
        z_cl0 = z_cav0 + p.xlr_cavity
        z_cl1 = z_cl0 + p.clamp_len
        R = p.mic_d / 2 + p.iso_thk
        s = cyl(Ds / 2, p.socket_depth + p.holder_floor, (0, 0, -p.socket_depth), (0, 0, 1))
        s = s.fuse(cyl(23, p.xlr_cavity, (0, 0, z_cav0), (0, 0, 1)))
        s = s.fuse(cq.Solid.makeCone(23, R + 5, 12, cq.Vector(0, 0, z_cl0 - 6), cq.Vector(0, 0, 1)))
        s = s.fuse(cyl(R + 5, p.clamp_len, (0, 0, z_cl0), (0, 0, 1)))
        lug_z0, lug_z1 = z_cl0 + 10, z_cl1 - 5
        s = s.fuse(box(-8, 8, R + 3, R + 13, lug_z0, lug_z1))
        s = s.cut(box(-sq / 2, sq / 2, -sq / 2, sq / 2, -p.socket_depth - 1, 0))
        s = s.cut(cyl(4.25 if P == 30 else 6.5, p.holder_floor + 2, (0, 0, -1), (0, 0, 1)))  # core bolt
        s = s.cut(cyl(18, p.xlr_cavity - 4, (0, 0, z_cav0), (0, 0, 1)))
        s = s.cut(cyl(R, p.clamp_len + 10, (0, 0, z_cl0 - 6), (0, 0, 1)))
        s = s.cut(box(-1, 1, 0, R + 14, z_cl0 + 4, z_cl1 + 1))  # clamp slit
        s = s.cut(cyl(2.2, 30, (-15, R + 8, (lug_z0 + lug_z1) / 2), (1, 0, 0)))  # M4 pinch bolt
        s = s.cut(box(-7, 7, -30, 0, z_cav0, z_cav0 + 34))  # cable exit + drain window
        return s, z_cl1

    # ------------------------------------------------------------------ assemblies
    def ground_mic(self, m):
        p, g, P = self.p, self.g, self.P
        d = V(*m["axis"])
        org = V(m["pos"][0], m["pos"][1], g.board_top)
        L = loc(org, d, Z)
        bh = p.board_size / math.sqrt(2)
        # board: square, edges at 45 deg to the mic axis (= world-aligned for corner mics)
        x1 = p.ac_from_tip - p.collar_from_tip - p.sleeve_flange
        xcr = x1 - p.clamp_len / 2
        b_holes = [(120.0, 0.0, self.pr["dowel"], "dowel"), (-60.0, 0.0, self.pr["dowel"], "dowel"),
                   (60.0, 0.0, self.pr["clear"], "csk"), (-200.0, 0.0, self.pr["clear"], "csk"),
                   (xcr, 25.0, 5.5, "m5"), (xcr, -25.0, 5.5, "m5")]
        key = "ground_board"
        if key not in self.plate_defs:
            sol = self.make_plate_local([(bh, 0), (0, bh), (-bh, 0), (0, -bh)], b_holes, p.board_thk)
            self.plate_defs[key] = dict(outline=[(bh, 0), (0, bh), (-bh, 0), (0, -bh)], holes=b_holes,
                                        material=f"{p.board_thk:.0f} mm HPL compact laminate (or 6 mm Al)",
                                        solid=sol, thk=p.board_thk)
        self.plate_count[key] += 1
        self.add("ground_board", self.plate_defs[key]["solid"].moved(loc(org - Z * p.board_thk, d, Z)), C_BOARD)
        self.hw_dowel(2)
        self.hw[f"{self.thr}x20 countersunk ISO 10642, A4"] += 2
        self.hw[f"T-nut {self.thr} for slot {self.pr['slot']:.0f} (hammer or drop-in), A2/A4 stainless"] += 2
        self.hw["M5x45 socket head ISO 4762, A4 + M5 nyloc + 2 washers (mic clamp)"] += 2

        base, cap, (cx0, cx1) = self.cradle_shapes()
        self.add("print_ground_cradle", base.moved(L), C_PRINT)
        self.add("print_ground_cap", cap.moved(L), C_PRINT)
        self.printed["ground_cradle"] += 1
        self.printed["ground_cap"] += 1
        ax = V(0, 0, g.axis_h)
        tipx = p.ac_from_tip
        rear = V(tipx - p.mic_len, 0, g.axis_h)
        self.add("mic", self.mic_shape(rear, (1, 0, 0)).moved(L), C_MIC)
        self.add("print_sleeve_tpu", self.sleeve_shape(V(cx0, 0, g.axis_h), (1, 0, 0)).moved(L), C_TPU)
        self.printed["iso_sleeve_tpu"] += 1
        xc = tipx - p.collar_from_tip
        col = self.collar_shape().moved(loc(V(xc, 0, g.axis_h), (0, 0, 1), (1, 0, 0)))
        self.add("print_index_collar", col.moved(L), C_PRINT)
        self.printed["index_collar"] += 1
        self.hw["M3x16 socket head + M3 nut, A4 (collar)"] += 1
        plug = cyl(10, p.xlr_plug_len, rear - V(p.xlr_plug_len, 0, 0), (1, 0, 0))
        self.add("xlr_plug", plug.moved(L), C_MIC)
        ws = cq.Solid.makeSphere(p.windscreen_d / 2, cq.Vector(0, 0, g.axis_h)).cut(
            box(-100, 100, -100, 100, -100, 0))
        self.add("windscreen_keepout", ws.moved(L), C_WIND)
        # strain relief on the outrigger just inboard of the board; leave a slack loop
        clip_org = V(-bh - 30, 0, g.z1 - g.board_top)
        self.add("print_cable_clip", self.clip_shape().moved(loc(clip_org, (1, 0, 0), Z)).moved(L), C_PRINT)
        self.printed["cable_clip"] += 1
        self.hw_tbolt(1, length=25)

    def apex(self):
        p, g, P = self.p, self.g, self.P
        xm = nrm(V(1, 1, 0))
        org = V(0, 0, g.mast_top)
        L = loc(org, xm, Z)
        holder, z_cl1 = self.apex_holder_shape()
        self.add("print_apex_holder", holder.moved(L), C_PRINT)
        self.printed["apex_holder"] += 1
        self.hw[f"{self.pr['core_tap']}x16 socket head ISO 4762, A4 + washer (apex holder -> mast)"] += 1
        self.hw["M4x30 socket head + M4 nyloc, A4 (apex clamp)"] += 1
        tip = V(0, 0, p.apex_height + p.ac_from_tip)
        rear = tip - V(0, 0, p.mic_len)
        self.add("mic", self.mic_shape(rear, (0, 0, 1)), C_MIC)
        clamp_top = org + V(0, 0, z_cl1)
        self.add("print_sleeve_tpu", self.sleeve_shape(clamp_top - V(0, 0, p.clamp_len), (0, 0, 1)), C_TPU)
        self.printed["iso_sleeve_tpu"] += 1
        self.add("print_index_collar", self.collar_shape().moved(
            cq.Location(cq.Vector(*(tip - V(0, 0, p.collar_from_tip))))), C_PRINT)
        self.printed["index_collar"] += 1
        self.hw["M3x16 socket head + M3 nut, A4 (collar)"] += 1
        self.add("xlr_plug", cyl(10, p.xlr_plug_len, rear - V(0, 0, p.xlr_plug_len), (0, 0, 1)), C_MIC)
        self.add("windscreen_keepout", cq.Solid.makeSphere(p.windscreen_d / 2, cq.Vector(0, 0, p.apex_height)), C_WIND)
        # strain relief clip on the mast below the holder (cable exits the holder window on -y side)
        dface = nrm(V(1, -1, 0))
        corg = dface * (P / 2) + V(0, 0, g.mast_top - p.socket_depth - 45)
        self.add("print_cable_clip", self.clip_shape().moved(loc(corg, Z, dface)), C_PRINT)
        self.printed["cable_clip"] += 1
        self.hw_tbolt(1, length=25)

    # ------------------------------------------------------------------ build
    def build(self):
        p, g, P = self.p, self.g, self.P
        a, zc, t = g.a, g.z0 + P / 2, p.plate_thk

        # ---- hub ----
        self.plate("hub_top", self.hub_def(True), V(0, 0, g.z1), V(1, 0, 0), Z)
        self.plate("hub_bottom", self.hub_def(False), V(0, 0, g.z0 - t), V(1, 0, 0), Z)
        self.foot(0, 0, g.z0 - t)

        # ---- corners: arms, outriggers, perimeter halves, gussets, struts ----
        corner_ang = {"NE": 0, "SE": -90, "SW": 180, "NW": 90}
        mast_xy = {}
        out_inner = p.end_gap_outrigger
        for idx, (name, (sx, sy)) in enumerate(AP.CORNERS.items()):
            d = V(sx * S2, sy * S2, 0)
            C = V(sx * a, sy * a, zc)
            perp = np.cross(Z, d)
            self.member("arm", d * p.hub_gap + V(0, 0, zc), C - d * p.end_gap_node, perp,
                        note="H7 through-hole 20 mm from outer end, 45 mm from hub end")
            self.member("outrigger", C + d * out_inner, C + d * (p.outrigger + 150), perp,
                        note=f"H7 through-holes 20 mm from inner end and at "
                             f"{p.outrigger - out_inner - 60:.0f} / {p.outrigger - out_inner + 120:.0f} mm (board dowels)")
            for e in (V(-sx, 0, 0), V(0, -sy, 0)):
                M = C + e * a
                self.member("side_half", C + e * p.end_gap_node, M - e * 1.5, np.cross(Z, e),
                            note="H7 through-holes 20 mm from each end")
            ang = math.radians(corner_ang[name])
            gx = V(math.cos(ang), math.sin(ang), 0)
            self.plate("corner_gusset", self.gusset_def(), V(C[0], C[1], g.z1), gx, Z)
            self.plate("corner_gusset", self.gusset_def(), V(C[0], C[1], g.z0 - t), gx, Z)
            self.foot(C[0], C[1], g.z0 - t)
            self.foot(*(C + d * (p.outrigger + 125))[:2], g.z0)
            # strut
            stag = 0.0 if idx % 2 == 0 else 50.0  # stagger so neighbouring mast clevises don't clash
            base_o = V(C[0], C[1], g.zg) - d * p.strut_base_inset
            self.clevis(base_o, d, Z)
            Apin = base_o + Z * p.pin_offset
            top_o = d * (P / 2) + V(0, 0, g.z_strut_top - stag)
            self.clevis(top_o, Z, d, mast=True)
            self.hw_dowel(1)
            self.hw_tbolt(2)
            Bpin = top_o + d * p.pin_offset
            u = nrm(Bpin - Apin)
            self.member(f"strut_{'A' if stag == 0 else 'B'}", Apin - u * 15, Bpin + u * 15, perp,
                        note="8 mm H8 cross-holes for clevis pins 15 mm from each end (pin-to-pin "
                             f"{np.linalg.norm(Bpin - Apin):.1f} mm)")

        # ---- mid-side nodes ----
        for sname, (sx, sy) in AP.SIDES.items():
            n = V(sx, sy, 0)
            M = n * a
            self.plate("midside_tee", self.tee_def(), V(M[0], M[1], g.z1), n, Z)
            self.plate("midside_tee", self.tee_def(), V(M[0], M[1], g.z0 - t), n, Z)
            self.foot(M[0], M[1], g.z0 - t)

        # ---- mast ----
        xm = nrm(V(1, 1, 0))
        self.member("mast", V(0, 0, g.mast_base), V(0, 0, g.mast_top), xm,
                    note=f"tap core {self.pr['core_tap']} both ends; H7 dowel holes at clevis heights "
                         f"{g.z_strut_top - g.mast_base:.0f} and {g.z_strut_top - 50 - g.mast_base:.0f} mm from foot")
        self.hw[f"{self.pr['core_tap']}x16 socket head ISO 4762, A4 + washer (mast -> hub)"] += 1
        for sx, sy in AP.CORNERS.values():
            d = V(sx * S2, sy * S2, 0)
            self.angle_bracket(d * (P / 2) + V(0, 0, g.zg), d, Z)
        # bracket bolts into hub are counted as hub plate holes; remove the double count
        self.hw_tbolt(-4)

        # ---- mic 6 (optional) ----
        if p.mic6:
            n = V(*AP.SIDES[p.mic6_side], 0)
            M = n * a + V(0, 0, zc)
            self.member("outrigger", M + n * (P / 2 + 5), M + n * (P / 2 + 5 + p.outrigger + 150 - out_inner),
                        np.cross(Z, n), note="(mic 6) same as corner outriggers")
            self.foot(*(M + n * (P / 2 + 5 + p.outrigger + 105))[:2], g.z0)

        # ---- ground mics + apex ----
        for m in g.mics:
            if m["kind"] == "ground":
                self.ground_mic(m)
        self.apex()

        # ---- electronics box ----
        if p.box_side:
            n = V(*AP.SIDES[p.box_side], 0)
            tng = np.cross(Z, n)
            M = n * a + V(0, 0, zc)
            bw, bd, bh = p.box_size
            depth = bd + 60
            for sgn in (1, -1):
                A0 = M - n * (P / 2 + 0.5) + tng * sgn * (bw / 2 + 20)
                self.member("box_crossbar", A0, A0 - n * depth, tng, note="no holes")
                self.foot(*(A0 - n * (depth - 25))[:2], g.z0)
                # corner brackets in the horizontal plane, one each side of the crossbar
                for s2 in (1, -1):
                    o = A0 + tng * s2 * (P / 2) - Z * 10
                    s = box(0, 28, 0, 4, 0, 20).fuse(box(0, 4, 0, 28, 0, 20))
                    # local x = along crossbar (inward), local y = away from the crossbar
                    yv = tng * s2
                    fr = cq.Plane(origin=tuple(map(float, o)), xDir=tuple(map(float, -n)),
                                  normal=tuple(map(float, np.cross(-n, yv))))
                    self.add("angle_bracket", s.moved(cq.Location(fr)), C_STEEL)
                    self.buy[f"Corner bracket {p.profile} (28x28 cast), with 2 bolts + T-nuts"] += 1
                    self.hw_tbolt(2)
            pw, pd = bw + 2 * 20 + P + 40, depth
            org = M + Z * (P / 2) - n * (P / 2 + 0.5)
            plate = box(-pw / 2, pw / 2, 0, pd, 0, 4)
            for sgn in (1, -1):
                for yy in (30, pd - 30):
                    plate = plate.cut(cyl(self.pr["clear"] / 2, 10, (sgn * (bw / 2 + 20), yy, -1), (0, 0, 1)))
            # xDir = tng = z x n, normal = z  ->  local y = z x tng = -n, i.e. inward
            fr = cq.Plane(origin=tuple(map(float, org)), xDir=tuple(map(float, tng)), normal=(0.0, 0.0, 1.0))
            self.add("box_plate", plate.moved(cq.Location(fr)), C_PLATE)
            self.buy[f"Al plate {pw:.0f} x {pd:.0f} x 4 mm (box mounting plate)"] += 1
            self.hw_tbolt(4)
            bx = box(-bw / 2, bw / 2, 30, 30 + bd, 4, 4 + bh)
            self.add("electronics_box", bx.moved(cq.Location(fr)), C_BOX)
            self.box_centre = org - n * (30 + bd / 2) + Z * (4 + bh / 2)
            self.buy[f"Weatherproof enclosure {bw:.0f}x{bd:.0f}x{bh:.0f} mm IP66 + cable glands"] += 1
            self.hw["M5x16 + washer A4 (box to plate)"] += 4

        # ---- pole bracket (optional) ----
        if p.pole_side:
            n = V(*AP.SIDES[p.pole_side], 0)
            tng = np.cross(Z, n)
            M = n * a + V(0, 0, zc)
            st0 = M + n * (P / 2 + 5)
            self.member("pole_stub", st0, st0 + n * 250, tng, note="H7 through-hole 20 mm from inner end")
            self.foot(*(st0 + n * 225)[:2], g.z0)
            post_c = st0 + n * 200
            self.member("pole_post", V(post_c[0], post_c[1], g.z1), V(post_c[0], post_c[1], g.z1 + p.post_len), n,
                        note="tap core both ends")
            for s2 in (1, -1):
                self.angle_bracket(V(post_c[0], post_c[1], g.z1) + tng * s2 * (P / 2), tng * s2, Z)
            pc = post_c + n * (P / 2 + p.pole_d / 2 + 6)
            self.add("pole", cyl(p.pole_d / 2, p.pole_len, (pc[0], pc[1], 0), (0, 0, 1)), C_POLE)
            for zz in (g.z1 + 120, g.z1 + p.post_len - 60):
                ring = cyl(p.pole_d / 2 + 4, 20, (pc[0], pc[1], zz), (0, 0, 1)).cut(
                    cyl(p.pole_d / 2, 22, (pc[0], pc[1], zz - 1), (0, 0, 1)))
                blk = box(-12, 12, -12, 12, 0, 20).moved(cq.Location(cq.Vector(
                    *(post_c + n * (P / 2 + 3))[:2], zz)))
                self.add("pole_clamp", ring.fuse(blk), C_STEEL)
            self.buy[f"U-bolt pole clamp for {p.pole_d:.0f} mm pole onto {p.profile} slot, A4"] += 2
            self.hw_tbolt(4)
            self.pole_xy = pc[:2]

        self.hw[f"Levelling foot {self.pr['foot']} thread, 50 mm pad, stainless stem"] += len(self.feet)

    # ------------------------------------------------------------------ outputs
    def export_step(self, path):
        assy = cq.Assembly(name="acoustic_array")
        for name, shape, col in self.parts:
            assy.add(shape, name=name, color=cq.Color(*col))
        assy.export(path) if hasattr(assy, "export") else assy.save(path)

    def export_printed(self, d):
        os.makedirs(d, exist_ok=True)
        base, cap, _ = self.cradle_shapes()
        holder, _ = self.apex_holder_shape()
        rx = cq.Location(cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), -90)  # +x axis -> +z
        out = {
            "ground_cradle": base,
            "ground_cap": cap.moved(cq.Location(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180)),  # print flat face down
            "iso_sleeve_tpu": self.sleeve_shape(V(0, 0, 0), (0, 0, 1)),
            "index_collar": self.collar_shape(),
            "cable_clip": self.clip_shape(),
            "apex_holder": holder.moved(cq.Location(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180)),  # socket up
        }
        for k, s in out.items():
            cq.exporters.export(cq.Workplane().add(s), os.path.join(d, f"{k}.stl"), tolerance=0.02, angularTolerance=0.1)
        return list(out)

    def export_dxf(self, d):
        os.makedirs(d, exist_ok=True)
        for key, pd in self.plate_defs.items():
            t = pd.get("thk", self.p.plate_thk)
            sec = cq.Workplane().add(pd["solid"]).section(t / 2)
            cq.exporters.export(sec, os.path.join(d, f"{key}.dxf"))

    def export_bom(self, path):
        p, g = self.p, self.g
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
        mat = {k: v["material"] for k, v in self.plate_defs.items()}
        for key, q in self.plate_count.items():
            pdef = self.plate_defs[key]
            t = pdef.get("thk", p.plate_thk)
            pts = np.array(pdef["outline"])
            ext = pts.max(0) - pts.min(0)
            if key == "ground_board":
                ext = (p.board_size, p.board_size)
            nd = sum(1 for h in pdef["holes"] if h[3] == "dowel")
            rows.append(("plate", key, f"{mat[key]}, t={t:.0f} mm, envelope {ext[0]:.0f} x {ext[1]:.0f} mm, "
                                       f"DXF plates/{key}.dxf", q,
                         f"{nd} dowel holes {self.pr['dowel']:.0f} H7 (ream), other holes per DXF"))
        for k, q in self.buy.items():
            rows.append(("bought", k, "", q, ""))
        for k, q in sorted(self.hw.items()):
            if q:
                rows.append(("fastener", k, "", q, "add ~10 % spares"))
        mats = {"ground_cradle": "ASA or PETG, 4 walls, 40 % infill",
                "ground_cap": "ASA or PETG, 4 walls, 40 % infill",
                "iso_sleeve_tpu": "TPU 85A-95A (or cut from EPDM tube 22x28 mm)",
                "index_collar": "ASA or PETG", "cable_clip": "ASA/PETG (TPU also works)",
                "apex_holder": "ASA or PETG, 5 walls, 50 % infill, print socket-up"}
        for k, q in self.printed.items():
            rows.append(("printed", k, mats.get(k, ""), q, f"stl/{k}.stl"))
        rows.append(("bought", "Foam windscreen + furry cover", f"~{p.windscreen_d:.0f} mm; ground mics: half-dome on board", 5 + int(p.mic6), ""))
        rows.append(("bought", "XLR cable, outdoor PUR", "length to suit box", 5 + int(p.mic6), ""))
        with open(path, "w") as f:
            f.write("category,item,spec,qty,notes\n")
            for r in rows:
                f.write(",".join('"' + str(c).replace('"', "'") + '"' for c in r) + "\n")
        return rows

    def export_render_meshes(self, d):
        """One STL per colour group (for the OpenSCAD preview renderer)."""
        os.makedirs(d, exist_ok=True)
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


def render(files, outdir, g, p):
    exe = shutil.which("openscad")
    if not exe:
        print("openscad not found; skipping renders")
        return []
    os.makedirs(outdir, exist_ok=True)
    scad = os.path.join(outdir, "assembly.scad")
    with open(scad, "w") as f:
        for fn, col in files:
            f.write(f'color([{col[0]},{col[1]},{col[2]},{col[3]}]) import("{fn}");\n')
    r = g.r_mic
    m0 = g.mics[0]["pos"]
    views = {
        "iso": (f"{2.6 * r:.0f},{-3.0 * r:.0f},{2.2 * r:.0f},0,0,{g.mast_top / 2.5:.0f}", "1800,1200"),
        "top": (f"0,0,{6 * r:.0f},0,0,0", "1400,1400"),
        "south_elevation": (f"0,{-6 * r:.0f},{g.mast_top / 2:.0f},0,0,{g.mast_top / 2:.0f}", "1800,1100"),
        "ground_mic_closeup": (f"{m0[0] + 450:.0f},{m0[1] - 650:.0f},{450:.0f},{m0[0] - 120:.0f},{m0[1] - 120:.0f},60", "1400,1000"),
        "apex_closeup": (f"{700:.0f},{-950:.0f},{p.apex_height + 200:.0f},0,0,{g.mast_top + 60:.0f}", "1000,1300"),
    }
    out = []
    for name, (cam, size) in views.items():
        png = os.path.join(outdir, f"{name}.png")
        ortho = name in ("top", "south_elevation")
        cmd = [exe, "-o", png, f"--imgsize={size}", f"--camera={cam}", "--colorscheme=Tomorrow",
               f"--projection={'o' if ortho else 'p'}"] + (["--viewall"] if ortho else []) + [scad]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-800:])
        out.append(png)
    return out


def parse_overrides(items):
    p = AP.DEFAULT
    types = {f.name: f.type for f in fields(AP.Params)}
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown parameter {k}")
        cur = getattr(p, k)
        if isinstance(cur, bool):
            kw[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, float):
            kw[k] = float(v)
        elif isinstance(cur, tuple):
            kw[k] = tuple(float(x) for x in v.split(","))
        else:
            kw[k] = v
    return replace(p, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help="override a Params field, e.g. --set apex_height=1800")
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
    rows = m.export_bom(os.path.join(OUT, "bom.csv"))

    # ---- sanity checks printed to the console ----
    mics = np.array([mm["pos"] for mm in g.mics])
    print(f"parts: {len(m.parts)}  printed STLs: {printed}")
    longest = max(pr["length"] for pr in m.profiles)
    print(f"longest profile piece: {longest:.0f} mm (limit {p.max_piece:.0f})")
    if hasattr(m, "box_centre"):
        dbox = np.linalg.norm(mics - m.box_centre, axis=1).min()
        print(f"electronics box centre -> nearest mic: {dbox:.0f} mm")
    if hasattr(m, "pole_xy"):
        dp = np.linalg.norm(mics[:, :2] - m.pole_xy, axis=1).min()
        print(f"pole -> nearest mic (horizontal): {dp:.0f} mm")
    print(f"feet: {len(m.feet)}")
    if not args.no_render:
        files = m.export_render_meshes(os.path.join(OUT, "render", "mesh"))
        pngs = render(files, os.path.join(OUT, "render"), g, p)
        print("renders:", [os.path.basename(x) for x in pngs])


if __name__ == "__main__":
    main()
