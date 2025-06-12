import typing as ty
import elodin as el
import jax
from dataclasses import field
import jax.numpy as jnp
import csv
import os


# utility function
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


Thrust = ty.Annotated[jax.Array, el.Component("thrust", el.ComponentType.F64, metadata={"priority": 2})]
Motor = ty.Annotated[jax.Array, el.Component("rocket_motor", el.ComponentType.F64)]

thrust_curve = parse(os.path.dirname(os.path.abspath(__file__))+"/data/AeroTech_M685W.txt")

@el.dataclass
class Rocket(el.Archetype):
  thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))
  motor: Motor = field(default_factory=lambda: jnp.float64(0.0))


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
