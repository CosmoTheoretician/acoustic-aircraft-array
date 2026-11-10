#!/usr/bin/env python3
"""
Export the array model for Blender: one binary STL per material class, plus a JSON scene
description that array_blender_render.py reads to place cameras and keep grass off the footings.

Units in the STL files are millimetres in the array frame (x east, y north, z up, z=0 = ground).
The Blender script scales them by 0.001 to metres. The wind-screen keep-out spheres are exported
as "foam": in the render they stand in for the real 90 mm foam windscreens.

Run:  .venv/bin/python array_blender_export.py [--set name=value ...]
Writes out/blender/<material>.stl (git-ignored) and out/blender/scene.json.
"""

import argparse
import collections
import json
import os

import cadquery as cq

import array_model as AM

# colour constant in array_model -> material class used by array_blender_render.py
MATERIALS = {
    AM.C_ALU: "aluminium",     # 40x40 / 80x40 extrusions, mast, solar stand frame
    AM.C_PLATE: "plate",       # laser-cut steel plates (galvanised / zinc-plated)
    AM.C_STEEL: "steel",       # stainless brackets, pins, fork heads, clamps
    AM.C_PRINT: "printed",     # ASA/PETG holders, collars, clips
    AM.C_TPU: "rubber",        # TPU sleeves
    AM.C_MIC: "mic",           # capsule bodies + XLR plugs
    AM.C_WIND: "foam",         # windscreens (keep-out volume in the CAD)
    AM.C_GROUND: "groundscrew",
    AM.C_BOX: "enclosure",
    AM.C_POLE: "pole",
    AM.C_PV: "pv",
    AM.C_SENSOR: "white",      # wind sensor, CPE210 bridge, sky camera
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help="override a Params field, e.g. --set apex_rise=1600")
    ap.add_argument("--tolerance", type=float, default=0.2, help="mesh chord tolerance, mm")
    args = ap.parse_args()
    p = AM.parse_overrides(args.set)
    m = AM.Model(p)
    m.build()
    g = m.g
    out = os.path.join(AM.OUT, "blender")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".stl"):
            os.remove(os.path.join(out, f))

    groups = collections.defaultdict(list)
    footings = []
    for n, s, c in m.parts:
        groups[MATERIALS[c]].append(s)
        if n.startswith("ground_screw"):
            bb = s.BoundingBox()
            footings.append([(bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2])
    files = {}
    for mat, shapes in groups.items():
        fn = f"{mat}.stl"
        cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound(shapes)), os.path.join(out, fn),
                            tolerance=args.tolerance, angularTolerance=0.05)
        files[mat] = fn

    def bbox(prefix):
        sh = [s for n, s, _ in m.parts if n.startswith(prefix)]
        if not sh:
            return None
        bb = cq.Compound.makeCompound(sh).BoundingBox()
        return [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax]

    scene = dict(
        files=files,
        mics=[list(map(float, mm["pos"])) for mm in g.mics],
        z_apex=float(g.z_apex), mast_top=float(g.mast_top),
        pole_xy=[float(v) for v in getattr(m, "pole_xy", (0, 0))], pole_len=float(p.pole_len),
        box_centre=[float(v) for v in m.box_centre],
        footings=footings,
        solar=bbox("solar_panel"), cpe=bbox("wifi_bridge_cpe"), wind_sensor=bbox("wind_sensor"),
    )
    with open(os.path.join(out, "scene.json"), "w") as fh:
        json.dump(scene, fh, indent=1)
    print(f"{len(files)} material groups, {len(footings)} ground screws -> {out}")


if __name__ == "__main__":
    main()
