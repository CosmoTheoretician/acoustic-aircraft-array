"""
Acoustic aircraft-tracking array: single source of truth for all parameters.

Everything (CAD model, BOM, mic coordinate file, DOA analysis, tracking simulation)
derives from the Params defined here, so the mic coordinates the tracker reads
always match the hardware.

Units: millimetres for every mechanical parameter.
Array coordinate system: origin at the centre of the base square at nominal
ground level, x = east, y = north, z = up. The base square's sides are aligned
with x/y, so the four ground mics sit on the diagonals (NE, SE, SW, NW).
"z = 0" is the plane `foot_height` below the underside of the base profiles,
i.e. the ground when the levelling feet are at their nominal setting.

Mic coordinates refer to the capsule's acoustic centre, taken to be
`ac_from_tip` behind the tip of the mic.
"""

from dataclasses import dataclass, asdict, replace, field
import hashlib
import json
import math

# --------------------------------------------------------------------------------------
# PARAMETERS — edit these.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    # ---- Array geometry (this is what DOA accuracy depends on) ----
    square_side: float = 2000.0  # ground-mic square side, capsule to capsule
    apex_height: float = 1500.0  # apex capsule acoustic centre above z=0

    # ---- Frame ----
    profile: str = "3030"  # "3030" or "4040" (see PROFILES below)
    max_piece: float = 1200.0  # longest single profile; longer members are spliced
    foot_height: float = 40.0  # ground -> underside of base profiles, feet at mid-travel
    plate_thk: float = 6.0  # gusset / hub / splice plates (aluminium 5083 or 6082)
    outrigger: float = 450.0  # corner node -> ground mic, along the diagonal
    end_gap_node: float = 70.0  # arm / side profile ends stop this far from a node centre
    end_gap_outrigger: float = 20.0  # outrigger starts this far out from the corner node
    hub_gap: float = 20.0  # arm ends stop this far from the hub centre
    pin_offset: float = 32.0  # strut clevis pin: height above gusset / distance out from mast face
    strut_base_inset: float = 45.0  # bottom strut pin sits this far inward of the corner node
    strut_top_drop: float = 60.0  # top strut pins this far below the mast top
    splice_len: float = 240.0  # splice plate length for split members

    # ---- Ground-plane boards ----
    board_size: float = 400.0  # square board, capsule over its centre
    board_thk: float = 8.0  # e.g. 8 mm HPL compact laminate or 6 mm aluminium

    # ---- Microphone. Behringer ECM8000 values from retailer listings — MEASURE YOURS ----
    mic_d: float = 22.0  # body diameter (listings say ~21 mm; 22 leaves room for paint/tolerance)
    mic_len: float = 195.0  # overall length excl. XLR plug (listings: 190-197 mm)
    mic_tip_d: float = 12.7  # capsule end diameter (Thomann lists 12.7 mm)
    mic_nose_len: float = 45.0  # taper length body -> tip (UNVERIFIED, estimated from photos)
    ac_from_tip: float = 3.0  # acoustic centre behind the tip (diaphragm just inside the grille)
    xlr_plug_len: float = 55.0  # female XLR plug incl. boot
    cable_d: float = 6.5

    # ---- Mic mounts (3D printed, PETG/ASA; sleeve in TPU 85A or EPDM tube) ----
    iso_thk: float = 3.0  # isolation sleeve wall
    sleeve_flange: float = 2.0  # sleeve flange = washer between index collar and mount
    collar_from_tip: float = 140.0  # index collar's front face distance from mic tip
    clamp_len: float = 45.0  # length of the clamped section
    cradle_floor: float = 2.0  # printed floor under the sleeve (ground mics)
    xlr_cavity: float = 90.0  # apex holder: room for the XLR plug + cable bend
    holder_floor: float = 6.0
    socket_depth: float = 40.0  # apex holder socket over the mast end
    windscreen_d: float = 90.0  # foam + furry cover; no structure inside this keep-out

    # ---- Options ----
    mic6: bool = True  # optional 6th ground mic on a mid-side outrigger
    mic6_side: str = "E"
    box_size: tuple = (300.0, 200.0, 150.0)  # electronics box W x D x H
    box_side: str = "S"
    pole_side: str = "W"  # Wi-Fi / camera pole bracket; "" to omit
    pole_d: float = 40.0
    pole_len: float = 2000.0
    post_len: float = 600.0


PROFILES = {
    # size, slot opening, centre bore, T-nut/bolt thread, dowel dia, bolt clearance
    "3030": dict(P=30.0, slot=8.2, bore=6.8, thread="M6", dowel=6.0, clear=6.6, foot="M8", core_tap="M8"),
    "4040": dict(P=40.0, slot=10.2, bore=10.5, thread="M8", dowel=8.0, clear=9.0, foot="M10", core_tap="M12"),
}

SIDES = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
CORNERS = {"NE": (1, 1), "SE": (1, -1), "SW": (-1, -1), "NW": (-1, 1)}

# --------------------------------------------------------------------------------------
# Derived geometry
# --------------------------------------------------------------------------------------


@dataclass
class Geometry:
    p: Params
    P: float = 0.0
    prof: dict = field(default_factory=dict)
    r_mic: float = 0.0  # ground mic radius from centre
    r_c: float = 0.0  # corner node radius
    a: float = 0.0  # half side of the frame square (node coords are (+-a, +-a))
    z0: float = 0.0  # underside of base profiles
    z1: float = 0.0  # top of base profiles
    zg: float = 0.0  # top of top gussets / hub plate
    board_top: float = 0.0
    axis_h: float = 0.0  # ground mic axis above board top
    z_ground_mic: float = 0.0
    mast_base: float = 0.0
    mast_top: float = 0.0
    mast_len: float = 0.0
    z_strut_top: float = 0.0
    mics: list = field(default_factory=list)


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def derive(p: Params) -> Geometry:
    g = Geometry(p=p)
    if p.profile not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    g.prof = PROFILES[p.profile]
    g.P = g.prof["P"]
    g.r_mic = p.square_side / math.sqrt(2)
    g.r_c = g.r_mic - p.outrigger
    if g.r_c < 300:
        raise ValueError("square too small for this outrigger length")
    g.a = g.r_c / math.sqrt(2)
    g.z0 = p.foot_height
    g.z1 = g.z0 + g.P
    g.zg = g.z1 + p.plate_thk
    g.board_top = g.z1 + p.board_thk
    g.axis_h = p.mic_d / 2 + p.iso_thk + p.cradle_floor
    g.z_ground_mic = g.board_top + g.axis_h

    # Apex: acoustic centre at apex_height. Stack (top-down): tip -> collar face
    # (collar_from_tip) -> sleeve flange -> clamp -> XLR cavity -> holder floor -> mast top.
    tip = p.apex_height + p.ac_from_tip
    clamp_top = tip - p.collar_from_tip - p.sleeve_flange
    g.mast_top = clamp_top - p.clamp_len - p.xlr_cavity - p.holder_floor
    g.mast_base = g.zg
    g.mast_len = g.mast_top - g.mast_base
    g.z_strut_top = g.mast_top - p.strut_top_drop
    if g.mast_len < 300:
        raise ValueError("apex_height too low for the mic + holder stack")

    s2 = 1 / math.sqrt(2)
    mics = []
    for i, (name, (sx, sy)) in enumerate(CORNERS.items(), start=1):
        d = (sx * s2, sy * s2, 0.0)
        mics.append(dict(id=i, name=name, kind="ground",
                         pos=(g.r_mic * d[0], g.r_mic * d[1], g.z_ground_mic),
                         axis=d))  # axis = direction the capsule points (tip outward)
    mics.append(dict(id=5, name="APEX", kind="apex", pos=(0.0, 0.0, p.apex_height), axis=(0.0, 0.0, 1.0)))
    if p.mic6:
        sx, sy = SIDES[p.mic6_side]
        r6 = g.a + p.outrigger
        mics.append(dict(id=6, name=f"MID_{p.mic6_side}", kind="ground",
                         pos=(sx * r6, sy * r6, g.z_ground_mic), axis=(float(sx), float(sy), 0.0)))
    g.mics = mics
    return g


def params_hash(p: Params) -> str:
    return hashlib.sha256(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest()[:12]


def mic_table(p: Params, include_optional=True):
    """Mic capsule positions in metres, array frame. Used by the analysis scripts."""
    g = derive(p)
    rows = []
    for m in g.mics:
        if not include_optional and m["id"] == 6:
            continue
        rows.append(dict(id=m["id"], name=m["name"], kind=m["kind"],
                         x=m["pos"][0] / 1000, y=m["pos"][1] / 1000, z=m["pos"][2] / 1000,
                         axis=m["axis"]))
    return rows


def mic_array_m(p: Params, include_optional=False):
    import numpy as np
    return np.array([[r["x"], r["y"], r["z"]] for r in mic_table(p, include_optional)])


def write_mic_files(p: Params, json_path, csv_path):
    g = derive(p)
    rows = mic_table(p)
    doc = {
        "description": "Acoustic array mic capsule positions (acoustic centres)",
        "frame": {"origin": "centre of base square at nominal ground level",
                  "x": "east", "y": "north", "z": "up", "units": "m"},
        "params_hash": params_hash(p),
        "params": asdict(p),
        "mics": [dict(id=r["id"], name=r["name"], kind=r["kind"],
                      position_m=[round(r["x"], 5), round(r["y"], 5), round(r["z"], 5)],
                      axis=[round(c, 6) for c in r["axis"]],
                      optional=r["id"] == 6) for r in rows],
        "notes": ["Positions are nominal design values. Survey the built array and/or "
                  "self-calibrate against ADS-B: 1 cm of position error = 29 us of timing error."],
    }
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)
    with open(csv_path, "w") as f:
        f.write("id,name,kind,x_m,y_m,z_m,axis_x,axis_y,axis_z\n")
        for r in rows:
            f.write(f"{r['id']},{r['name']},{r['kind']},{r['x']:.5f},{r['y']:.5f},{r['z']:.5f},"
                    f"{r['axis'][0]:.6f},{r['axis'][1]:.6f},{r['axis'][2]:.6f}\n")
    return doc


DEFAULT = Params()

if __name__ == "__main__":
    g = derive(DEFAULT)
    for k in ("r_mic", "r_c", "a", "z1", "zg", "board_top", "z_ground_mic", "mast_top", "mast_len"):
        print(f"{k:14s} {getattr(g, k):9.1f} mm")
    for m in g.mics:
        print(m["name"], [round(c, 1) for c in m["pos"]])
