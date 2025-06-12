
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
import elodin as el
import polars as pl

def euler_to_quat(angles: jax.Array) -> el.Quaternion:
    [roll, pitch, yaw] = jnp.deg2rad(angles)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return el.Quaternion(jnp.array([x, y, z, w]))

# coverts `val` to a coordinate along some series `s`
def to_coord(s: pl.Series, val: jax.Array) -> jax.Array:
    s_min = s.min()
    s_max = s.max()
    s_count = len(s.unique())
    return (val - s_min) * (s_count - 1) / jnp.clip(s_max - s_min, 1e-06)

def quat_from_vecs(v1: jax.Array, v2: jax.Array) -> el.Quaternion:
    v1 = v1 / la.norm(v1)
    v2 = v2 / la.norm(v2)
    n = jnp.cross(v1, v2)
    w = jnp.dot(v2, v2) * jnp.dot(v1, v1) + jnp.dot(v1, v2)
    q = el.Quaternion.from_array(jnp.array([*n, w])).normalize()
    return q