"""
3.5 m S-band / L-band dish extension: single source of truth for all parameters.

Purpose: Doppler velocimetry of lunar spacecraft (Intuitive Machines IM-3, LRO as rehearsal)
at S-band (2200-2290 MHz), and radio astronomy (21 cm hydrogen line, Sun, bright sources)
at L-band. Receive only.

Mechanics: commercial aluminium mesh reflector on our own elevation-over-azimuth
positioner. The el axis runs behind the backframe (so nothing sits inside the bowl), and a
counterweight balances dish + feed about it. Both axes use worm-gear slewing drives (solar-tracker type, self-locking),
each driven by a 48 V servo motor through a planetary gearbox, with an absolute encoder on
the output. The positioner stands on a steel pedestal tube on a concrete footing.

Units: millimetres, kilograms, unless a name says otherwise.
Dish frame: origin at the pedestal axis at ground level, x = east, y = north, z = up.
Values marked UNVERIFIED come from typical catalogue data and must be checked against the
parts actually bought.
"""

from dataclasses import dataclass, asdict
import math

C = 299_792_458.0


@dataclass(frozen=True)
class DishParams:
    # ---- Reflector (commercial mesh kit) ----
    D: float = 3500.0  # aperture diameter
    f_over_D: float = 0.40  # typical for mesh kits; feeds are designed for this
    reflector_mass: float = 45.0  # kit incl. ribs, hub, mesh (UNVERIFIED, ask the vendor)
    mesh_opening: float = 6.0  # mesh aperture; < lambda/20 at 2.3 GHz keeps leakage < 0.1 dB
    surface_rms: float = 4.0  # kit surface accuracy, mm RMS (UNVERIFIED; Ruze loss uses this)
    aperture_eff: float = 0.55  # illumination x spillover x blockage x phase (typical prime focus)
    backframe_depth: float = 200.0  # hub / backframe behind the vertex

    # ---- Feeds (quick-swap, dowel-located plate at the focus) ----
    feed_mass: float = 4.0  # feed + LNA box
    feed_support_mass: float = 6.0  # quadripod + mount plate
    quadripod_r: float = 1600.0  # strut feet radius on the reflector
    s_feed: tuple = (110.0, 260.0)  # S-band dual-CP septum feed: diameter x length (envelope)
    l_feed: tuple = (160.0, 330.0)  # L-band circular waveguide feed with choke (envelope)

    # ---- Positioner geometry ----
    el_axis_height: float = 2300.0  # elevation axis above ground
    axis_behind_vertex: float = 500.0  # el axis behind the vertex: hub + plate clear the el drive and its worm
    hub_d: float = 400.0  # backframe hub diameter (stays clear of the yoke upright in x)
    yoke_height: float = 1000.0  # az drive top flange -> el axis (room for the counterweight at zenith)
    el_drive_x: float = -150.0  # el drive centre along the el axis; the yoke upright is outboard of it
    cw_arm: float = 600.0  # counterweight centre behind the el axis
    pedestal_d: float = 219.1  # steel tube, 219.1 x 6.3 (S355), hot-dip galvanised
    pedestal_t: float = 6.3
    min_rim_clearance: float = 400.0  # rim above ground at el = 0
    park_az: float = 180.0  # render pose: pointing (az from north, el)
    park_el: float = 35.0

    # ---- Drives (UNVERIFIED catalogue values for SE-series slewing drives) ----
    az_drive: str = "SE9"
    el_drive: str = "SE7"
    gearbox_ratio: float = 50.0  # planetary between servo and worm
    worm_eff: float = 0.30  # self-locking worms run at ~25-35 % efficiency
    servo_rated_torque: float = 1.27  # Nm, 400 W 48 V integrated servo (3000 rpm)
    slew_rate_dps: float = 1.0  # deg/s at the output (stow from anywhere in < 4 min)
    backlash_deg: float = 0.15  # slewing-drive backlash at the output

    # ---- Wind (EN 1991-1-4 style; check the national annex) ----
    vb0: float = 26.0  # basic wind speed, m/s (Luxembourg NA value to be checked)
    terrain_z0: float = 0.3  # m, terrain category III (suburban / garden with trees)
    v_operate: float = 12.0  # m/s mean wind: above this the tracker stows
    cf_face: float = 0.65  # drag coefficient, mesh dish facing the wind (solid ~1.4)
    cf_stow_h: float = 0.20  # horizontal force coefficient, stowed at zenith (on D^2 pi/4)
    cf_stow_v: float = 0.35  # vertical (lift) coefficient, stowed at zenith
    cm_el: float = 0.10  # aerodynamic moment coefficient about the el axis (on q A D)
    cm_az: float = 0.06  # about the az axis

    # ---- Foundation ----
    footing: tuple = (1300.0, 1300.0, 1000.0)  # concrete block L x W x H
    concrete_density: float = 2400.0  # kg/m3
    fos_overturn: float = 1.5

    # ---- RF ----
    f_s_mhz: float = 2210.6  # IM-1 observed carrier; IM-3 to be taken from its FCC filing
    f_lro_mhz: float = 2271.2  # LRO S-band downlink
    f_hi_mhz: float = 1420.405752
    lna_nf_db_s: float = 0.5
    lna_nf_db_l: float = 0.4
    feed_loss_db: float = 0.2  # feed + filter loss ahead of the LNA (filter after the LNA)
    t_spill_k: float = 20.0  # ground spillover seen by a ~-12 dB edge taper at f/D 0.4
    t_sky_k: float = 8.0  # atmosphere + CMB + galaxy at moderate elevation

    # ---- Electronics ----
    gpsdo_adev_1s: float = 1e-11  # Allan deviation of the GPS-disciplined reference at 1 s

    def to_dict(self):
        return asdict(self)


# UNVERIFIED typical SE-series data (Chinese / Kinematics-type slewing drives)
DRIVES = {
    "SE7": dict(ratio=73, out_torque=1.5e3, hold_torque=13.5e3, tilt_moment=13.5e3, mass=23.0,
                housing_d=280.0, height=120.0),
    "SE9": dict(ratio=61, out_torque=4.3e3, hold_torque=38.7e3, tilt_moment=33.9e3, mass=33.0,
                housing_d=340.0, height=130.0),
    "SE12": dict(ratio=78, out_torque=6.5e3, hold_torque=43.0e3, tilt_moment=54.0e3, mass=52.0,
                 housing_d=430.0, height=140.0),
}


class DishGeometry:
    def __init__(self, p: DishParams):
        self.p = p
        self.f = p.f_over_D * p.D  # focal length
        self.depth = p.D ** 2 / (16 * self.f)
        self.area = math.pi * p.D ** 2 / 4 * 1e-6  # m2
        # rim half-angle seen from the focus (for feed taper)
        self.psi0 = 2 * math.degrees(math.atan(p.D / (4 * self.f)))
        self.z_az = p.el_axis_height - p.yoke_height  # az drive top flange
        self.az = DRIVES[p.az_drive]
        self.el = DRIVES[p.el_drive]
        self.pedestal_top = self.z_az - self.az["height"]
        # mass properties about the el axis, along the boresight (+ = towards the focus)
        a = p.axis_behind_vertex
        zr = a + self.depth / 3  # reflector CG ~ depth/3 in front of the vertex
        zf = a + self.f
        zs = a + 0.6 * self.f  # quadripod
        self.m_dish = p.reflector_mass + p.feed_mass + p.feed_support_mass
        self.moment_dish = (p.reflector_mass * zr + p.feed_mass * zf + p.feed_support_mass * zs) / 1000  # kg m
        self.cw_mass = self.moment_dish / (p.cw_arm / 1000)  # balances the dish about the el axis
        self.m_moving = self.m_dish + self.cw_mass
        self.cg_el = 0.0  # balanced by design
        self.rim_clear_el0 = p.el_axis_height - p.D / 2
        # at zenith the counterweight hangs below the el axis: clearance to the az turntable
        self.zenith_cw_gap = p.yoke_height - p.cw_arm - 120 - 20
        if self.rim_clear_el0 < p.min_rim_clearance:
            raise ValueError(f"rim only {self.rim_clear_el0:.0f} mm above ground at el=0")
        if self.zenith_cw_gap < 100:
            raise ValueError("counterweight hits the turntable at zenith: raise yoke_height or shorten cw_arm")

    def gain_dbi(self, f_mhz):
        lam = C / (f_mhz * 1e6) * 1e3  # mm
        ruze = math.exp(-(4 * math.pi * self.p.surface_rms / lam) ** 2)
        return 10 * math.log10(self.p.aperture_eff * ruze * (math.pi * self.p.D / lam) ** 2)

    def hpbw_deg(self, f_mhz):
        lam = C / (f_mhz * 1e6) * 1e3
        return 70.0 * lam / self.p.D  # ~-12 dB edge taper


DEFAULT = DishParams()

if __name__ == "__main__":
    g = DishGeometry(DEFAULT)
    print(f"f = {g.f:.0f} mm, depth = {g.depth:.0f} mm, rim angle from focus = {g.psi0:.1f} deg")
    print(f"dish + feed {g.m_dish:.0f} kg, moment {g.moment_dish:.1f} kg m about the el axis -> counterweight "
          f"{g.cw_mass:.0f} kg at {DEFAULT.cw_arm:.0f} mm; moving mass {g.m_moving:.0f} kg")
    print(f"rim above ground at el 0: {g.rim_clear_el0:.0f} mm; counterweight gap at zenith: {g.zenith_cw_gap:.0f} mm")
    for f in (DEFAULT.f_hi_mhz, DEFAULT.f_s_mhz):
        print(f"{f:8.1f} MHz: gain {g.gain_dbi(f):.1f} dBi, HPBW {g.hpbw_deg(f):.2f} deg")
