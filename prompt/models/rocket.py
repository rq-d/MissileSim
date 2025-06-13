import typing as ty
import elodin as el
import jax
from dataclasses import field
import jax.numpy as jnp
import jax.numpy.linalg as la
import csv
import os
import polars as pl
from jax.scipy.ndimage import map_coordinates
from  utils import math

# constants
thrust_vector_body_frame = jnp.array([-1.0, 0.0, 0.0])
a_ref = 24.89130 / 100**2 #  reference area from cfd
l_ref = 5.43400 / 100
xmc = 0.40387
SIM_TIME_STEP = 1.0 / 120.0
lp_sample_freq = round(1.0 / SIM_TIME_STEP)
lp_buffer_size = lp_sample_freq * 4
lp_cutoff_freq = 1
PRONAVGAIN = 5
pitch_pid = [0.01, 0.02,1.8]
pitch_pid = [0.0,0.0,0.0]
# pitch_pid = [0.05, 0.01,0.0]

LANDATX = -12000 # TODO MAKE THIS AN INPUT THIS IS TEMPORARY TARGET LOCATION

aero_df = pl.from_dict({
    'Mach': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'Alphac': [0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0],
    'Delta': [-40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0],
    'CmR': [-5.997, -6.905, -8.235, -10.83, -5.315, -6.008, -5.918, -5.714, 0.0, 1.313, 2.335, 0.4163, 5.315, 3.642, 2.977, 1.061, 5.997, 5.372, 4.191, 1.882, -7.269, -8.373, -9.873, -12.93, -6.323, -7.255, -7.14, -6.846, 0.0, 1.486, 2.681, 0.445, 6.323, 4.263, 3.463, 1.222, 7.269, 6.396, 4.963, 2.27, -11.53, -12.49, -13.88, -15.71, -9.056, -8.891, -8.448, -8.155, 0.0, 1.921, 3.144, 1.169, 9.056, 8.419, 7.126, 4.228, 11.53, 10.14, 8.19, 4.94],
    'CA': [1.121, 1.028, 0.9495, 0.9803, 0.6405, 0.5852, 0.4342, 0.217, 0.2942, 0.2873, 0.2591, 0.2032, 0.6405, 0.5988, 0.635, 0.6333, 1.121, 1.215, 1.246, 1.267, 1.242, 1.137, 1.051, 1.095, 0.6902, 0.6278, 0.4588, 0.2184, 0.2924, 0.2856, 0.2577, 0.2025, 0.6902, 0.6434, 0.6895, 0.6967, 1.242, 1.351, 1.392, 1.425, 1.851, 1.747, 1.621, 1.48, 0.9888, 0.8509, 0.658, 0.4269, 0.448, 0.4446, 0.4345, 0.418, 0.9888, 1.06, 1.111, 1.154, 1.851, 1.961, 2.03, 2.098],
    'CZR': [-1.092, -0.3878, 0.3984, 1.141, -1.141, -0.4069, 0.7324, 2.176, 0.0, 1.061, 2.368, 3.494, 1.141, 1.561, 2.483, 3.64, 1.092, 1.789, 2.577, 3.68, -1.191, -0.4161, 0.4355, 1.252, -1.274, -0.4526, 0.8073, 2.408, 0.0, 1.178, 2.63, 3.88, 1.274, 1.736, 2.755, 4.043, 1.191, 1.973, 2.844, 4.07, -1.609, -0.8494, 0.1373, 1.323, -1.639, -0.5395, 0.9159, 2.704, 0.0, 1.304, 2.894, 4.443, 1.639, 2.532, 3.576, 4.981, 1.609, 2.483, 3.481, 4.811]
})  # fmt: skip

# TODO fix pathing and put this in utilities
def parse(filename):
  data = {
      "time": [],
      "thrust": []
  }

  # Replace with your actual file name
  with open(filename, mode='r', newline='') as csvfile:
      reader = csv.reader(csvfile)
      
      for row in reader:
          if len(row) != 2:
              continue  # Skip malformed rows
          
          time_val = row[0].strip()
          thrust_val = row[1].strip()
          
          data["time"].append(float(time_val))
          data["thrust"].append(float(thrust_val))
  return data
thrust_curve = parse(os.path.dirname(os.path.abspath(__file__))+"/data/AeroTech_M685W.txt")

def second_order_butterworth(
    signal: jax.Array, f_sampling: int, f_cutoff: int, method: str = "forward"
) -> jax.Array:
    "https://stackoverflow.com/questions/20924868/calculate-coefficients-of-2nd-order-butterworth-low-pass-filter"
    if method == "forward_backward":
        signal = second_order_butterworth(signal, f_sampling, f_cutoff, "forward")
        return second_order_butterworth(signal, f_sampling, f_cutoff, "backward")
    elif method == "forward":
        pass
    elif method == "backward":
        signal = jnp.flip(signal, axis=0)
    else:
        raise NotImplementedError
    ff = f_cutoff / f_sampling
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita**2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita**2) * b0

    def f(carry, x_i):
        x_im1, x_im2, y_im1, y_im2 = carry
        y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
        return (x_i, x_im1, y_i, y_im1), y_i

    init = (signal[1], signal[0]) * 2
    signal = jax.lax.scan(f, init, signal[2:])[1]
    signal = jnp.concatenate((signal[0:1],) * 2 + (signal,))
    if method == "backward":
        signal = jnp.flip(signal, axis=0)
    return signal


def aero_interp_table(df: pl.DataFrame) -> jax.Array:
    coefs = ["CmR", "CA", "CZR"]
    aero = jnp.array(
        [
            [
                df.group_by(["Alphac"], maintain_order=True)
                .agg(pl.col(coefs).min())
                .select(pl.col(coefs))
                .to_numpy()
                for _, df in df.group_by(["Delta"], maintain_order=True)
            ]
            for _, df in df.group_by(["Mach"], maintain_order=True)
        ]
    )
    aero = aero.transpose(3, 0, 1, 2)
    return aero

Thrust = ty.Annotated[jax.Array, el.Component("thrust", el.ComponentType.F64, metadata={"priority": 17})]
Motor = ty.Annotated[jax.Array, el.Component("rocket_motor", el.ComponentType.F64)]
AeroForce = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "aero_force",
        el.ComponentType.SpatialMotionF64,
        metadata={"element_names": "τx,τy,τz,x,y,z"},
    ),
]
AeroCoefs = ty.Annotated[
    jax.Array,
    el.Component(
        "aero_coefs",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "Cl,CnR,CmR,CA,CZR,CYR"},
    ),
]
CenterOfGravity = ty.Annotated[jax.Array, el.Component("center_of_gravity", el.ComponentType.F64)]
DynamicPressure = ty.Annotated[jax.Array, el.Component("dynamic_pressure", el.ComponentType.F64)]
Mach = ty.Annotated[jax.Array, el.Component("mach", el.ComponentType.F64)]
AngleOfAttack = ty.Annotated[jax.Array, el.Component("angle_of_attack", el.ComponentType.F64)]
FinDeflect = ty.Annotated[jax.Array, el.Component("fin_deflect", el.ComponentType.F64)]
FinControl = ty.Annotated[jax.Array, el.Component("fin_control", el.ComponentType.F64)]
Wind = ty.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
AccelSetpointSmooth = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint_smooth",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 100},
    ),
]
AccelSetpoint = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 101},
    ),
]
ProNavSetpoint = ty.Annotated[
    jax.Array,
    el.Component(
        "pronav_setpoint",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 100},
    ),
]
VRelAccel = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 20},
    ),
]

VRelAccelBuffer = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel_buffer",
        el.ComponentType(el.PrimitiveType.F64, (lp_buffer_size, 3)),
        metadata={"priority": -1},
    ),
]

VRelAccelFiltered = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel_filtered",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 19},
    ),
]
PitchPID = ty.Annotated[
    jax.Array,
    el.Component(
        "pitch_pid",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "Kp,Ki,Kd"},
    ),
]

PitchPIDState = ty.Annotated[
    jax.Array,
    el.Component(
        "pitch_pid_state",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "e,i,d", "priority": 18},
    ),
]

@el.dataclass
class Rocket(el.Archetype):
  thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))
  motor: Motor = field(default_factory=lambda: jnp.float64(0.0))
  aero_force: AeroForce = field(default_factory=lambda: el.SpatialForce())
  aero_coefs: AeroCoefs = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
  center_of_gravity: CenterOfGravity = field(default_factory=lambda: jnp.float64(0.2)) # real CG is 0.74 from SW data
  dynamic_pressure: DynamicPressure = field(default_factory=lambda: jnp.float64(0.0))
  mach: Mach = field(default_factory=lambda: jnp.float64(0.0))
  angle_of_attack: AngleOfAttack = field(default_factory=lambda: jnp.array([0.0]))
  fin_deflect: FinDeflect = field(default_factory=lambda: jnp.float64(0.0))
  fin_control: FinControl = field(default_factory=lambda: jnp.float64(0.0))
  wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
  
  v_rel_accel_buffer: VRelAccelBuffer = field(
        default_factory=lambda: jnp.zeros((lp_buffer_size, 3))
    )
  v_rel_accel: VRelAccel = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
  v_rel_accel_filtered: VRelAccelFiltered = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
  accel_setpoint: AccelSetpoint = field(default_factory=lambda: jnp.array([0.0, 0.0]))
  accel_setpoint_smooth: AccelSetpointSmooth = field(default_factory=lambda: jnp.array([0.0, 0.0]))
  pronav_setpoint: ProNavSetpoint = field(
        default_factory=lambda: jnp.array([0.0,0.0])
    )
  pitch_pid: PitchPID = field(default_factory=lambda: jnp.array(pitch_pid))
  pitch_pid_state: PitchPIDState = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
  

# --------------Models Related to the Rocket--------------


# Effector --- this component adds forces to the current sumforce due to aerodynamic effects to 
@el.map
def apply_aero_forces(
    p: el.WorldPos,
    f_aero: AeroForce,
    f: el.Force,
) -> el.Force:
    # convert from body frame to world frame
    return f + p.angular() @ f_aero

@el.map
def pitch_pid_state(
    a_setpoint: ProNavSetpoint,
    a_rel: VRelAccelFiltered,
    s: PitchPIDState,
) -> PitchPIDState:
    # a_setpoint = jnp.array([0,0]) # test code
    e = a_rel[2] - a_setpoint[0]
    i = jnp.clip(s[1] + e * SIM_TIME_STEP * 2, -2.0, 2.0)
    d = e - s[0]
    pid_state = jnp.array([e, i, d])
    return pid_state

@el.map
def pitch_pid_control(pid: PitchPID, s: PitchPIDState) -> FinControl:
    Kp, Ki, Kd = pid
    e, i, d = s
    mv = Kp * e + Ki * i + Kd * d
    fin_control = mv * SIM_TIME_STEP
    return fin_control

# ProNav works on y and z axes but PID code is implemented y
@el.map
def pronav_setpoint(accel: ProNavSetpoint, p: el.WorldPos, v:el.WorldVel) -> ProNavSetpoint:

    # target
    r_t = jnp.array([LANDATX, 0.0, 1.0]) 
    v_t = jnp.array([0,0,0])

    # missile
    r_m = jnp.array([p.linear()[0], p.linear()[1], p.linear()[2]])
    v_m = jnp.array([v.linear()[0], v.linear()[1], v.linear()[2]])

    v_r = v_t - v_m
    a = PRONAVGAIN *jnp.linalg.norm(v_r) * v_m / jnp.linalg.norm(v_m)
    r=r_t-r_m
    b = jnp.cross(r,v_r) / jnp.dot(r,r)
    res = jnp.cross(a,b)
    dist2target = jnp.linalg.norm(r)
    accel = jnp.array([res[0],res[1]])

    # applies ProNav when distance is close to target
    return jax.lax.cond(
        # dist2target < 7000.0,    # condition to be met
        False, # always on
        lambda _: accel,   # if true return value
        lambda _: jnp.array([0.0,0.0]),    # if false return valuec
        operand=None,
    )


@el.map
def v_rel_accel_buffer(a_rel: VRelAccel, buffer: VRelAccelBuffer) -> VRelAccelBuffer:
    return jnp.concatenate((buffer[1:], a_rel.reshape(1, 3)))

@el.map
def v_rel_accel(v: el.WorldVel, a: el.WorldAccel) -> VRelAccel:
    v = jax.lax.cond(
        la.norm(v.linear()) < 1e-6,
        lambda _: thrust_vector_body_frame,
        lambda _: v.linear(),
        operand=None,
    )
    v_rot = math.quat_from_vecs(thrust_vector_body_frame, v)
    a_rel = v_rot.inverse() @ a.linear()
    return a_rel

@el.map
def v_rel_accel_filtered(s: VRelAccelBuffer) -> VRelAccelFiltered:
    return second_order_butterworth(s, lp_sample_freq, lp_cutoff_freq)[-1]

@el.map
def accel_setpoint_smooth(a: AccelSetpoint, a_s: AccelSetpointSmooth) -> AccelSetpointSmooth:
    dt = SIM_TIME_STEP
    exp_decay_constant = 0.5
    return a_s + (a - a_s) * jnp.exp(-exp_decay_constant * dt)

@el.map
def fin_control(fd: FinDeflect, fc: FinControl, mach: Mach) -> FinDeflect:
    fc = fc / (0.1 + mach)
    fc = jnp.clip(fc, -0.2, 0.2)
    fd += fc
    fd = jnp.clip(fd, -40.0, 40.0)
    return fd

@el.map
def angle_of_attack(p: el.WorldPos, v: el.WorldVel, w: Wind) -> AngleOfAttack:
    # u = freestream velocity vector in body frame
    u = p.angular().inverse() @ (v.linear() - w)

    # angle of attack is the angle between the freestream velocity vector and the attitude vector
    angle_of_attack = jnp.dot(u, thrust_vector_body_frame) / jnp.clip(la.norm(u), 1e-6)
    angle_of_attack = jnp.rad2deg(jnp.arccos(angle_of_attack)) * -jnp.sign(u[2])
    return angle_of_attack

@el.map
def mach(p: el.WorldPos, v: el.WorldVel, w: Wind) -> tuple[Mach, DynamicPressure]:
    atmosphere = {
        "h": jnp.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]),
        "T": jnp.array([15.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.2]),
        "p": jnp.array([101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.]),
        "d": jnp.array([1.225, 0.3639, 0.0880, 0.0132, 0.0014, 0.0009, 0.0001, 0.]),
    }  # fmt: skip
    altitude = p.linear()[2]
    temperature = jnp.interp(altitude, atmosphere["h"], atmosphere["T"]) + 273.15
    density = jnp.interp(altitude, atmosphere["h"], atmosphere["d"])
    specific_heat_ratio = 1.4
    specific_gas_constant = 287.05
    speed_of_sound = jnp.sqrt(specific_heat_ratio * specific_gas_constant * temperature)
    local_flow_velocity = la.norm(v.linear() - w)
    mach = local_flow_velocity / speed_of_sound
    dynamic_pressure = 0.5 * density * local_flow_velocity**2
    dynamic_pressure = jnp.clip(dynamic_pressure, 1e-6)
    return mach, dynamic_pressure

# Non Effector ---This updates mach and dynamic pressure
@el.map
def aero_coefs(
    mach: Mach,
    angle_of_attack: AngleOfAttack,
    fin_deflect: FinDeflect,
) -> AeroCoefs:
    aero = aero_interp_table(aero_df)
    aoa_sign = jax.lax.cond(
        jnp.abs(angle_of_attack) < 1e-6,
        lambda _: 1.0,
        lambda _: jnp.sign(angle_of_attack),
        operand=None,
    )
    # aoa_sign is used to negate fin deflection angle as a way to interpolate negative angles of attack
    fin_deflect *= aoa_sign
    coords = [
        math.to_coord(aero_df["Mach"], mach),
        math.to_coord(aero_df["Delta"], fin_deflect),
        math.to_coord(aero_df["Alphac"], jnp.abs(angle_of_attack)),
    ]
    coefs = jnp.array([map_coordinates(coef, coords, 1, mode="nearest") for coef in aero])
    coefs = jnp.array(
        [
            0.0,
            0.0,
            coefs[0] * aoa_sign,
            coefs[1],
            coefs[2] * aoa_sign,
            0.0,
        ]
    )
    return coefs

@el.map
def aero_forces(
    aero_coefs: AeroCoefs,
    xcg: CenterOfGravity,
    q: DynamicPressure,
) -> AeroForce:
    Cl, CnR, CmR, CA, CZR, CYR = aero_coefs

    # shift CmR, CnR from MC to CG
    CmR = CmR - CZR * (xcg - xmc) / l_ref
    CnR = CnR - CYR * (xcg - xmc) / l_ref

    f_aero_linear = jnp.array([CA, CYR, CZR]) * q * a_ref 
    f_aero_torque = jnp.array([Cl, -CmR, CnR]) * q * a_ref * l_ref
    f_aero = el.SpatialForce(linear=f_aero_linear, torque=f_aero_torque)
    return f_aero

# Updates the thrust force by querying thrust curve dict
@el.system
def thrust(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    q: el.Query[Motor],
) -> el.Query[Thrust]:
    t = tick[0] * dt[0]
    time = jnp.array(thrust_curve["time"])
    thrust = jnp.array(thrust_curve["thrust"])
    f_t = jnp.interp(t, time, thrust)
    return q.map(Thrust, lambda _: f_t)

# Applies the thrust force using el.SpatialForce
@el.map
def apply_thrust(thrust: Thrust, f: el.Force, p: el.WorldPos) -> el.Force:    
  return f + el.SpatialForce(linear=p.angular() @ jnp.array([-1.0, 0.0, 0.0]) * thrust )
