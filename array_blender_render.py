"""
Photoreal Cycles renders of the acoustic array, from the meshes written by array_blender_export.py.

Run (Blender 4.2+ / 5.x, headless):
  .venv/bin/python array_blender_export.py
  /Applications/Blender.app/Contents/MacOS/Blender -b -P array_blender_render.py
  options after "--":  --views hero,northwest,mic,apex,cpe,solar,dusk   --samples 256   --res 1920x1080
                       --preview   (quarter resolution, 16 samples: for framing checks)
Writes out/blender/render_<view>.jpg

Background: real sky + landscape panoramas (CC0 HDRIs from Poly Haven, git-ignored, see HDRI
below) light the scene and fill the horizon. Near the array the ground is modelled: 3D grass
blades out to ~20 m, then a textured lawn that turns transparent between 60 and 150 m, so the
panorama's own meadow takes over towards the horizon. The panorama is rotated so its real sun
sits where each view wants it. Without the HDRI files the script falls back to a procedural
gradient sky + sun lamp.

Materials: clear-anodised aluminium extrusions (fine brushed roughness along the profile),
zinc-plated steel plates (spangle noise), stainless pins and brackets, orange ASA prints (matte,
slight layer-line bump), black TPU sleeves, dark foam windscreens (velvet sheen + coarse bump),
galvanised ground screws and pole, light-grey enclosure, glass-covered PV panel with a cell
grid, white plastic for the wind sensor, CPE210 and sky camera. Bevels + smoothing by angle
as in the dish renderer. Cycles on the Apple GPU (Metal), adaptive sampling + OIDN, AgX.
"""

import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))

# CC0 panoramas from polyhaven.com (download into assets/hdri/, git-ignored):
#   https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/8k/meadow_2_8k.hdr     ("Meadow 2", S. Majboroda)
#   https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/8k/lilienstein_8k.hdr  ("Lilienstein", A. Mischok)
HDRI_DIR = os.path.join(HERE, "assets", "hdri")
HDRI = {"day": "meadow_2_8k.hdr", "dusk": "lilienstein_8k.hdr"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = dict(views="hero,northwest,mic,apex,cpe,solar,dusk", samples=256, res="1920x1080", preview=False,
             grass=1500, exposure=0.0)
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k == "preview":
            a["preview"] = True
        else:
            a[k] = argv[i + 1]
            i += 1
        i += 1
    a["samples"] = int(a["samples"])
    a["grass"] = float(a["grass"])
    a["exposure"] = float(a["exposure"])
    return a


# ----------------------------------------------------------------------------- materials
def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bsdf, out


def setin(bsdf, name, value):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = value


def noise(nt, scale, detail=4.0, coords="Object"):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = detail
    nt.links.new(tc.outputs[coords], nz.inputs["Vector"])
    return nz


def ramp(nt, src, lo, hi):
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].color = lo
    r.color_ramp.elements[1].color = hi
    nt.links.new(src, r.inputs["Fac"])
    return r


def bump(nt, bsdf, height_src, strength, distance=0.002):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = distance
    nt.links.new(height_src, b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])


def mat_metal(name, base, rough, rough_var=0.0, var_scale=60.0, metallic=1.0, coat=0.0):
    m, nt, bsdf, _ = node_mat(name)
    setin(bsdf, "Base Color", (*base, 1))
    setin(bsdf, "Metallic", metallic)
    setin(bsdf, "Roughness", rough)
    setin(bsdf, "Coat Weight", coat)
    if rough_var:
        nz = noise(nt, var_scale, 6.0)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = rough - rough_var
        mr.inputs["To Max"].default_value = rough + rough_var
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return m


def mat_galvanised():
    # hot-dip galvanising: zinc spangle = patchy roughness + slight tint variation
    m, nt, bsdf, _ = node_mat("galvanised")
    setin(bsdf, "Metallic", 1.0)
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 90.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    r = ramp(nt, vor.outputs["Distance"], (0.56, 0.58, 0.60, 1), (0.70, 0.71, 0.72, 1))
    nt.links.new(r.outputs["Color"], bsdf.inputs["Base Color"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.28
    mr.inputs["To Max"].default_value = 0.48
    nt.links.new(vor.outputs["Distance"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return m


def mat_grass(haze=None, fade=None):
    """Lawn; with haze=(colour, strength) it fades into the sky colour with distance
    (aerial perspective), so the horizon is soft instead of a hard green/blue edge."""
    m, nt, bsdf, out = node_mat("lawn")
    big = noise(nt, 0.08, 3.0)
    small = noise(nt, 9.0, 6.0)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["Factor"].default_value = 0.5
    r1 = ramp(nt, big.outputs["Fac"], (0.10, 0.19, 0.05, 1), (0.20, 0.30, 0.08, 1))
    r2 = ramp(nt, small.outputs["Fac"], (0.08, 0.15, 0.04, 1), (0.26, 0.34, 0.10, 1))
    nt.links.new(r1.outputs["Color"], mix.inputs[6])
    nt.links.new(r2.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    setin(bsdf, "Roughness", 0.88)
    blades = noise(nt, 900.0, 2.0)
    bump(nt, bsdf, blades.outputs["Fac"], 0.6, 0.01)
    if haze:
        col, strength = haze
        cam = nt.nodes.new("ShaderNodeCameraData")
        k = nt.nodes.new("ShaderNodeMath")
        k.operation = "MULTIPLY"
        k.inputs[1].default_value = -1.0 / 350.0  # haze e-folding distance, m
        nt.links.new(cam.outputs["View Distance"], k.inputs[0])
        ex = nt.nodes.new("ShaderNodeMath")
        ex.operation = "EXPONENT"
        nt.links.new(k.outputs[0], ex.inputs[0])
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(ex.outputs[0], inv.inputs[1])
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*col, 1)
        em.inputs["Strength"].default_value = strength
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(inv.outputs[0], mix.inputs["Fac"])
        nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
        nt.links.new(em.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    if fade:  # (near, far) in metres: opaque lawn -> transparent, revealing the panorama's ground
        cam = nt.nodes.new("ShaderNodeCameraData")
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = fade
        nt.links.new(cam.outputs["View Distance"], mr.inputs["Value"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mr.outputs["Result"], mix.inputs["Fac"])
        nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
        nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def mat_plastic(name, base, rough, layer_bump=0.0, sheen=0.0):
    m, nt, bsdf, _ = node_mat(name)
    setin(bsdf, "Base Color", (*base, 1))
    setin(bsdf, "Roughness", rough)
    if sheen:
        setin(bsdf, "Sheen Weight", sheen)
        setin(bsdf, "Sheen Roughness", 0.6)
    if layer_bump:  # FDM layer lines, 0.2 mm pitch along z
        tc = nt.nodes.new("ShaderNodeTexCoord")
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.wave_type = "BANDS"
        wv.bands_direction = "Z"
        wv.inputs["Scale"].default_value = 5000.0
        nt.links.new(tc.outputs["Object"], wv.inputs["Vector"])
        bump(nt, bsdf, wv.outputs["Fac"], layer_bump, 0.0002)
    return m


def mat_foam():
    # open-cell PU foam: dark, no specular, velvet sheen, coarse cell bump
    m, nt, bsdf, _ = node_mat("foam")
    setin(bsdf, "Base Color", (0.035, 0.035, 0.04, 1))
    setin(bsdf, "Roughness", 1.0)
    setin(bsdf, "Specular IOR Level", 0.1)
    setin(bsdf, "Sheen Weight", 0.35)
    setin(bsdf, "Sheen Roughness", 0.8)
    setin(bsdf, "Sheen Tint", (0.25, 0.25, 0.27, 1))
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 900.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    bump(nt, bsdf, vor.outputs["Distance"], 0.8, 0.001)
    return m


def mat_aluminium():
    # clear-anodised 6063 extrusion: satin, anisotropic-looking streaks along the profile axis
    m, nt, bsdf, _ = node_mat("aluminium")
    setin(bsdf, "Base Color", (0.80, 0.81, 0.83, 1))
    setin(bsdf, "Metallic", 1.0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 200.0
    nz.inputs["Detail"].default_value = 3.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.26
    mr.inputs["To Max"].default_value = 0.40
    nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
    return m


def mat_pv():
    """PV module: AR glass over dark mono cells; a brick texture draws the cell gaps + busbars."""
    m, nt, bsdf, _ = node_mat("pv")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.0
    br.inputs["Scale"].default_value = 1.0 / 0.166  # 166 mm cells (in metres, object space)
    br.inputs["Mortar Size"].default_value = 0.012
    br.inputs["Brick Width"].default_value = 1.0
    br.inputs["Row Height"].default_value = 1.0
    br.inputs["Color1"].default_value = (0.015, 0.02, 0.045, 1)
    br.inputs["Color2"].default_value = (0.02, 0.025, 0.055, 1)
    br.inputs["Mortar"].default_value = (0.55, 0.56, 0.6, 1)
    nt.links.new(tc.outputs["Object"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    setin(bsdf, "Roughness", 0.08)
    setin(bsdf, "Coat Weight", 1.0)
    setin(bsdf, "Coat Roughness", 0.03)
    return m


def make_materials():
    return {
        "aluminium": mat_aluminium(),
        "plate": mat_galvanised(),
        "steel": mat_metal("steel", (0.74, 0.74, 0.75), 0.22, 0.06, var_scale=80.0),
        "printed": mat_plastic("printed", (0.80, 0.26, 0.04), 0.55, layer_bump=0.15),
        "rubber": mat_plastic("rubber", (0.02, 0.02, 0.02), 0.75),
        "mic": mat_metal("mic", (0.16, 0.16, 0.17), 0.35, 0.05, metallic=0.85),
        "foam": mat_foam(),
        "groundscrew": mat_metal("groundscrew", (0.48, 0.49, 0.50), 0.55, 0.12, var_scale=30.0),
        "enclosure": mat_plastic("enclosure", (0.72, 0.73, 0.70), 0.45),
        "pole": mat_metal("pole", (0.62, 0.63, 0.64), 0.42, 0.10, var_scale=12.0),
        "pv": mat_pv(),
        "white": mat_plastic("white", (0.82, 0.82, 0.80), 0.40),
    }


# ----------------------------------------------------------------------------- geometry
def import_stl(path, name):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path, global_scale=0.001)
    else:  # pre-4.1 Python importer
        bpy.ops.import_mesh.stl(filepath=path, global_scale=0.001)
    ob = (set(bpy.data.objects) - before).pop()
    ob.name = name
    return ob


def clean_and_smooth(ob, angle_deg=30.0, bevel_m=0.0):
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    lim = math.radians(angle_deg)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) == 2:
            e.smooth = e.calc_face_angle(0.0) < lim
        else:
            e.smooth = False
    bm.to_mesh(me)
    bm.free()
    if bevel_m > 0:
        bv = ob.modifiers.new("bevel", "BEVEL")
        bv.width = bevel_m
        bv.segments = 2
        bv.limit_method = "ANGLE"
        bv.angle_limit = lim
        try:
            bv.harden_normals = True
        except AttributeError:
            pass


def build_scene(sc, mats):
    d = os.path.join(HERE, "out", "blender")
    for mat, fn in sc["files"].items():
        ob = import_stl(os.path.join(d, fn), mat)
        ob.data.materials.clear()
        ob.data.materials.append(mats[mat])
        bevel = {"aluminium": 0.0008, "plate": 0.0006, "foam": 0.0, "pv": 0.0, "printed": 0.0006,
                 "mic": 0.0, "rubber": 0.0}.get(mat, 0.0005)
        clean_and_smooth(ob, 30.0, bevel)
    bpy.ops.mesh.primitive_plane_add(size=4000.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "lawn"
    ground.data.materials.append(mats["lawn"])
    return d


def world_sky(horizon, zenith, strength=1.0):
    w = bpy.data.worlds.new("sky")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.05
    mr.inputs["From Max"].default_value = 0.7
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].color = (*horizon, 1)
    r.color_ramp.elements[1].color = (*zenith, 1)
    nt.links.new(mr.outputs["Result"], r.inputs["Fac"])
    nt.links.new(r.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def sun(az_deg, el_deg, strength, color, angle_deg=0.6):
    data = bpy.data.lights.new("sun", "SUN")
    data.energy = strength
    data.color = color
    data.angle = math.radians(angle_deg)
    ob = bpy.data.objects.new("sun", data)
    bpy.context.collection.objects.link(ob)
    a, e = math.radians(az_deg), math.radians(el_deg)
    to_sun = Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))
    ob.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()


def camera(loc, target, lens, fstop=None):
    data = bpy.data.cameras.new("cam")
    data.lens = lens
    data.clip_start = 0.05
    data.clip_end = 5000
    ob = bpy.data.objects.new("cam", data)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if fstop:
        data.dof.use_dof = True
        data.dof.focus_distance = d.length
        data.dof.aperture_fstop = fstop
    bpy.context.scene.camera = ob


# ----------------------------------------------------------------------------- background
def hdri_sun(path, img):
    """Sun direction in the panorama (texture space): brightest pixel. Cached next to the file."""
    cache = path + ".sun.json"
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    lum = px.reshape(h, w, 4)[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    r, c = np.unravel_index(int(np.argmax(lum)), lum.shape)
    # Cycles equirectangular: u = 0.5 - atan2(y, x) / 2pi, rows run bottom -> top
    res = dict(phi=(0.5 - (c + 0.5) / w) * 2 * math.pi, el=((r + 0.5) / h - 0.5) * math.pi)
    with open(cache, "w") as fh:
        json.dump(res, fh)
    return res


def world_hdri(path, sun_az_deg, strength=1.0):
    """Panorama as world light + background, rotated so its sun is at azimuth sun_az_deg
    (from north through east). Returns the sun elevation of the panorama in degrees."""
    img = bpy.data.images.load(path, check_existing=True)
    sun = hdri_sun(path, img)
    theta = sun["phi"] - math.radians(90.0 - sun_az_deg)
    w = bpy.data.worlds.new("hdri")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.vector_type = "POINT"
    mp.inputs["Rotation"].default_value = (0.0, 0.0, theta)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    env.interpolation = "Cubic"
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return math.degrees(sun["el"])


def mat_blade():
    m, nt, bsdf, out = node_mat("grass_blade")
    oi = nt.nodes.new("ShaderNodeObjectInfo")  # per-instance random value in Cycles
    r = ramp(nt, oi.outputs["Random"], (0.045, 0.13, 0.025, 1), (0.20, 0.30, 0.07, 1))
    r.color_ramp.elements.new(0.85).color = (0.32, 0.33, 0.12, 1)  # a few dry blades
    nt.links.new(r.outputs["Color"], bsdf.inputs["Base Color"])
    setin(bsdf, "Roughness", 0.5)
    tl = nt.nodes.new("ShaderNodeBsdfTranslucent")
    nt.links.new(r.outputs["Color"], tl.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.3
    nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    nt.links.new(tl.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return m


def make_blade():
    """One grass blade, 1 m tall (scaled per instance to 4.5-11 cm), tapering, leaning forward."""
    verts, faces, n = [], [], 5
    for i in range(n + 1):
        t = i / n
        hw = 0.022 * (1 - t) + 0.002
        verts += [(-hw, 0.30 * t * t, t), (hw, 0.30 * t * t, t)]
    for i in range(n):
        a = 2 * i
        faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new("blade")
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat_blade())
    ob = bpy.data.objects.new("blade", me)
    bpy.context.collection.objects.link(ob)
    ob.hide_render = True  # only its instances render
    return ob


def _sock(sockets, name, typ):
    return next(s for s in sockets if s.name == name and s.type == typ and s.enabled) \
        if any(s.name == name and s.type == typ and s.enabled for s in sockets) \
        else next(s for s in sockets if s.name == name and s.type == typ)


def add_grass(radius, density, holes, hole_r=0.12):
    """Instanced 3D grass on a grid (so the density field can vary): fades out towards
    `radius`, none within hole_r of the ground-screw heads at `holes` (x, y in m)."""
    blade = make_blade()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=240, y_subdivisions=240, size=2 * radius, location=(0, 0, 0))
    area = bpy.context.active_object
    area.name = "grass"
    ng = bpy.data.node_groups.new("grass", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links.new
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.distribute_method = "RANDOM"
    dist.inputs["Density"].default_value = density
    L(gin.outputs[0], dist.inputs["Mesh"])
    pos = N.new("GeometryNodeInputPosition")
    ln = N.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    L(pos.outputs[0], ln.inputs[0])
    mr = N.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.65 * radius, radius
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0, 0.0
    L(ln.outputs["Value"], mr.inputs["Value"])
    inside = None
    for hx, hy in holes:
        dv = N.new("ShaderNodeVectorMath")
        dv.operation = "DISTANCE"
        dv.inputs[1].default_value = (hx, hy, 0.0)
        L(pos.outputs[0], dv.inputs[0])
        lt = N.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        lt.inputs[1].default_value = hole_r
        L(dv.outputs["Value"], lt.inputs[0])
        if inside is None:
            inside = lt
        else:
            mx = N.new("ShaderNodeMath")
            mx.operation = "MAXIMUM"
            L(inside.outputs[0], mx.inputs[0])
            L(lt.outputs[0], mx.inputs[1])
            inside = mx
    keep = N.new("ShaderNodeMath")
    keep.operation = "SUBTRACT"
    keep.inputs[0].default_value = 1.0
    L(inside.outputs[0], keep.inputs[1])
    fac = N.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    L(mr.outputs["Result"], fac.inputs[0])
    L(keep.outputs[0], fac.inputs[1])
    dens = N.new("ShaderNodeMath")  # Blender 5.x: Density is a field input in RANDOM mode
    dens.operation = "MULTIPLY"
    dens.inputs[1].default_value = density
    L(fac.outputs[0], dens.inputs[0])
    if dist.inputs["Density"].enabled and "Density Factor" not in [x.name for x in dist.inputs if x.enabled]:
        L(dens.outputs[0], dist.inputs["Density"])
    else:  # Blender 4.x: constant Density x field Density Factor
        L(fac.outputs[0], dist.inputs["Density Factor"])
    inst = N.new("GeometryNodeInstanceOnPoints")
    L(dist.outputs["Points"], inst.inputs["Points"])
    oi = N.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = blade
    L(oi.outputs["Geometry"], inst.inputs["Instance"])
    rr = N.new("FunctionNodeRandomValue")
    rr.data_type = "FLOAT_VECTOR"
    _sock(rr.inputs, "Min", "VECTOR").default_value = (-0.30, -0.30, 0.0)
    _sock(rr.inputs, "Max", "VECTOR").default_value = (0.30, 0.30, 2 * math.pi)
    rot_out = _sock(rr.outputs, "Value", "VECTOR")
    try:
        e2r = N.new("FunctionNodeEulerToRotation")
        L(rot_out, e2r.inputs[0])
        L(e2r.outputs[0], inst.inputs["Rotation"])
    except RuntimeError:
        L(rot_out, inst.inputs["Rotation"])
    rs = N.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    _sock(rs.inputs, "Min", "VALUE").default_value = 0.045
    _sock(rs.inputs, "Max", "VALUE").default_value = 0.11
    rs.inputs["Seed"].default_value = 7
    L(_sock(rs.outputs, "Value", "VALUE"), inst.inputs["Scale"])
    L(inst.outputs["Instances"], gout.inputs[0])
    mod = area.modifiers.new("grass", "NODES")
    mod.node_group = ng


# ----------------------------------------------------------------------------- views
def views(sc):
    mic = [Vector(m) / 1000 for m in sc["mics"]]
    se = mic[1]                       # south-east base mic
    apex = mic[4]
    cpe = sc["cpe"]
    cpe_c = Vector(((cpe[0] + cpe[3]) / 2, (cpe[1] + cpe[4]) / 2, (cpe[2] + cpe[5]) / 2)) / 1000
    so = sc["solar"]
    sol_c = Vector(((so[0] + so[3]) / 2, (so[1] + so[4]) / 2, (so[2] + so[5]) / 2)) / 1000
    day = dict(sky=((0.60, 0.72, 0.90), (0.10, 0.27, 0.72), 0.55))
    gold = dict(sky=((0.98, 0.70, 0.46), (0.22, 0.32, 0.62), 0.75))
    sunc = (1.0, 0.94, 0.85)
    return {
        # whole installation from the south-east: array, pole, solar stand
        "hero": dict(env="day", sun=(150, 35, 5.0, sunc), **day,
                     cam=((6.4, -8.6, 1.75), (0.0, -1.6, 1.85), 40, None)),
        # from the north-west: pole with CPE210 / wind sensor in front of the array
        "northwest": dict(env="day", sun=(300, 35, 5.0, sunc), **day,
                          cam=((-5.6, 5.6, 2.2), (0.0, -0.9, 2.0), 40, None)),
        # SE base mic: windscreen, printed holder, outrigger end
        "mic": dict(env="day", sun=(140, 35, 5.0, sunc), **day,
                    cam=(tuple(se + Vector((0.95, -1.05, 0.25))), tuple(se + Vector((0.0, 0.0, -0.04))), 70, 4.0)),
        # apex capsule and mast head
        "apex": dict(env="day", sun=(150, 40, 5.0, sunc), **day,
                     cam=(tuple(apex + Vector((0.85, -0.95, 0.05))), tuple(apex + Vector((0.0, 0.0, -0.12))), 50, 4.0)),
        # pole top: wind sensor, CPE210, sky camera
        "cpe": dict(env="day", sun=(220, 35, 5.0, sunc), **day,
                    cam=(tuple(cpe_c + Vector((-1.2, -1.4, 0.0))), tuple(cpe_c + Vector((0.0, 0.0, 0.33))), 45, 4.0)),
        # solar stand from the south, array behind
        "solar": dict(env="day", sun=(170, 40, 5.0, sunc), **day,
                      cam=(tuple(sol_c + Vector((2.6, -4.2, 0.2))), tuple(sol_c + Vector((-0.2, 0.8, 0.2))), 35, None)),
        # low evening sun from the west
        "dusk": dict(env="dusk", sun=(255, 7, 4.5, (1.0, 0.66, 0.40)), **gold,
                     cam=((7.4, -5.6, 1.3), (0.0, -1.4, 1.9), 38, None)),
    }


def setup_render(a):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    cyc = scn.cycles
    try:  # Apple GPU via Metal
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for dv in prefs.devices:
            dv.use = True
        cyc.device = "GPU"
    except Exception as e:  # noqa: BLE001
        print("GPU not available, rendering on CPU:", e)
    w, hgt = (int(v) for v in a["res"].split("x"))
    scn.render.resolution_x, scn.render.resolution_y = w, hgt
    scn.render.resolution_percentage = 25 if a["preview"] else 100
    cyc.samples = 16 if a["preview"] else a["samples"]
    cyc.use_adaptive_sampling = True
    cyc.adaptive_threshold = 0.01
    cyc.use_denoising = True
    try:
        cyc.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cyc.max_bounces = 8
    cyc.transparent_max_bounces = 48
    cyc.film_exposure = 1.0
    scn.view_settings.exposure = a["exposure"]
    try:
        scn.view_settings.view_transform = "AgX"
        scn.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scn.view_settings.view_transform = "Filmic"
    print("colour:", scn.view_settings.view_transform, "/", scn.view_settings.look)
    scn.render.image_settings.file_format = "JPEG"
    scn.render.image_settings.quality = 92


def main():
    a = parse_args()
    d = os.path.join(HERE, "out", "blender")
    with open(os.path.join(d, "scene.json")) as fh:
        sc = json.load(fh)
    allv = views(sc)
    for name in a["views"].split(","):
        v = allv[name]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        setup_render(a)
        mats = make_materials()
        horizon, zenith, strength = v["sky"]
        hdri = os.path.join(HDRI_DIR, HDRI[v["env"]])
        use_hdri = os.path.exists(hdri)
        mats["lawn"] = mat_grass(fade=(60.0, 150.0)) if use_hdri else mat_grass(haze=(horizon, strength))
        build_scene(sc, mats)
        if use_hdri:
            el = world_hdri(hdri, v["sun"][0])
            print(f"{name}: panorama {HDRI[v['env']]}, sun at az {v['sun'][0]} el {el:.1f}")
            add_grass(20.0, a["grass"] / (10 if a["preview"] else 1),
                      [(x / 1000, y / 1000) for x, y in sc["footings"]])
        else:
            print(f"{name}: {hdri} not found, procedural sky")
            world_sky(horizon, zenith, strength)
            sun(*v["sun"])
        camera(*v["cam"])
        out = os.path.join(d, f"render_{name}{'_preview' if a['preview'] else ''}.jpg")
        bpy.context.scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
