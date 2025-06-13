import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
import polars as pl
from jax.scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import numpy as np
import models.rocket
import utils.math as math
import models.models 
# import models.rocket
SIM_TIME_STEP = 1.0 / 120

def world(takeOffPitch: float = 70.0, missileMass: float = 16.0, targetX = -12000.0) -> el.World:
  w = el.World()
  ball_mesh = w.insert_asset(el.Mesh.sphere(0.4))
  ball_color = w.insert_asset(el.Material.color(12.7, 9.2, 0.5))
  
  rocket = w.spawn(
    [
      el.Body(
        world_pos=el.SpatialTransform(
        angular=math.euler_to_quat(jnp.array([0,takeOffPitch,0.0])), #initial orientation
        linear=jnp.array([0.0, 0.0, 0.0]),
        ),
        # inertia=el.SpatialInertia(missileMass, jnp.array([5, 200.00, 20.00])), 
        inertia=el.SpatialInertia(missileMass, jnp.array([0.1, 4, 4])), 
        # inertia=el.SpatialInertia(3.0, jnp.array([0.1, 1.0, 1.0])),
      ),
      models.rocket.Rocket(),
      w.glb("https://storage.googleapis.com/elodin-assets/rocket.glb"),   
    ],
    name = "Rocket",
  )

  target = w.spawn(
    [
      el.Body(
        world_pos=el.SpatialTransform(linear=jnp.array([targetX, 0.0, 0.01])),
        world_vel=el.SpatialTransform(el.WorldVel(linear=jnp.array([0.0, 0.0, 0.0])))
          ), 
      el.Shape(ball_mesh, ball_color),
      # Target(),
    ],
    name="target",
)


  # ------------ Visualize 
  w.spawn(
        el.Panel.hsplit(
            el.Panel.vsplit(
                el.Panel.viewport(
                    track_entity=rocket,
                    track_rotation=False,
                    pos=[5.0, 0.0, 1.0],
                    looking_at=[0.0, 0.0, 0.0],
                    show_grid=True,
                ),
                el.Panel.viewport(
                    track_entity=target,
                    track_rotation=False,
                    pos=[5.0, 0.0, 1.0],
                    looking_at=[0.0, 0.0, 0.0],
                    show_grid=True,
                ),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(rocket, models.rocket.FinDeflect)),
                el.Panel.graph(el.GraphEntity(rocket, models.rocket.FinControl)),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(rocket, models.rocket.ProNavSetpoint)),
                el.Panel.graph(el.GraphEntity(rocket, models.rocket.Thrust)),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(rocket, models.rocket.AngleOfAttack)),
                el.Panel.graph(el.GraphEntity(rocket, el.WorldAccel)),
            ),
            el.Panel.vsplit(
                el.Panel.graph(el.GraphEntity(rocket, models.rocket. Mach)),
            ),
            active=True,
        )
    )
  return w

def system() -> el.System:
    non_effectors = (models.models.ground
                    | models.rocket.mach
                    | models.rocket.angle_of_attack
                    | models.rocket.accel_setpoint_smooth
                    | models.rocket.v_rel_accel
                    | models.rocket.v_rel_accel_buffer
                    | models.rocket.v_rel_accel_filtered
                    | models.rocket.pronav_setpoint
                    | models.rocket.pitch_pid_state
                    | models.rocket.pitch_pid_control
                    | models.rocket.fin_control
                    | models.rocket.aero_coefs
                    | models.rocket.aero_forces
                    | models.rocket.thrust

 

    )
    effectors = models.models.gravity | models.rocket.apply_thrust | models.rocket.apply_aero_forces
    sys = non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.Rk4)
    return sys
