import elodin as el
import jax
import jax.numpy as jnp

@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())

# keeps entities from falling into earth
# taken from bounce example, although it doesnt work super well, its good enough for now
@el.map
def ground(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[2], v.linear()[2]) < 0.0,
        lambda _: el.SpatialMotion(linear=v.linear() * jnp.array([1.0, 1.0, -1.0]) * 0.001),
        lambda _: v,
        operand=None,
    )

