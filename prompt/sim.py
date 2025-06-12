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

SIM_TIME_STEP = 1.0 / 60.0



def world(takeOffPitch: float = 45.0) -> el.World:
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
        inertia=el.SpatialInertia(3.0, jnp.array([0.1, 1.0, 1.0])),
      ),
      models.rocket.Rocket(),
      # w.glb("https://storage.googleapis.com/elodin-assets/aim.glb"),
      # w.glb("https://storage.googleapis.com/elodin-assets/rocket.glb"),
      el.Shape(ball_mesh, ball_color),
      # models.rocket.Rocket(),
    ],
    name = "Rocket",
  )
  target = w.spawn(
            [
                el.Body(world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 1.0]))), # moved closer 100 meters
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
        active=True,
      )
    )
  )
  return w

def system() -> el.System:
    non_effectors = models.models.ground | models.rocket.thrust
    effectors = models.models.gravity | models.rocket.apply_thrust
    sys = non_effectors | el.six_dof(sys=effectors)
    return sys
