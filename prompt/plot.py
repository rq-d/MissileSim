import argparse

import elodin as el
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import polars as pl
from sim import system, world
import random

parser = argparse.ArgumentParser()
args = parser.parse_args()

# seed = random.seed(1)
randpitch = random.randint(45,70)
# exec = world(takeOffPitch=randpitch).build(system())
exec = world(takeOffPitch=68.26872848822481, targetX=-12000.0).build(system())

exec.run(10000)

# get missile data
df = exec.history("world_pos", el.EntityId(1))
df = df.with_columns(
    pl.col("world_pos").arr.get(4).alias("x"),
    pl.col("world_pos").arr.get(5).alias("y"),
    pl.col("world_pos").arr.get(6).alias("z"),
)
# get target data
df_tgt = exec.history("world_pos", el.EntityId(2))
df_tgt = df_tgt.with_columns( # add an alias
    pl.col("world_pos").arr.get(4).alias("x"),
    pl.col("world_pos").arr.get(5).alias("y"),
    pl.col("world_pos").arr.get(6).alias("z"),
)

# -------------------PLOTTING-----------------------
#
#
#

distance = np.linalg.norm(df.select(["x", "y", "z"]).to_numpy(), axis=1)
df = df.with_columns(pl.Series(distance).alias("distance"))
ticks = np.arange(df.shape[0])

# Print Final Position
# Print miss distance
missD = np.linalg.norm(df.select(["x", "y", "z"])[-1].to_numpy() - df_tgt.select(["x", "y", "z"])[-1].to_numpy(), axis=1)
print("\n\tMiss Distance (not perfect): \t", missD, " meters")
print("\tFinal Missile Position:\t\t" , df.select(["x", "y", "z"])[-1].to_numpy())
print("\tFinal Target Position:\t\t" , df_tgt.select(["x", "y", "z"])[-1].to_numpy())
print("")

plt.figure()
plt.plot(ticks, df["distance"])
plt.xlabel('Ticks')
plt.ylabel('Distance (m)')
plt.title('Missile Distance From Start')

plt.figure()
plt.plot(ticks, df.select(["z"]))
plt.xlabel('Ticks (s)')
plt.ylabel('Meters')
plt.title('z Pos')

# 3d Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(df.select(["x"]), df.select(["y"]), df.select(["z"]), 'black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Line Plot')



plt.figure()
plt.plot(ticks, exec.history("angle_of_attack", el.EntityId(1)))
plt.title('alpha')

# plt.figure()
# df = exec.history("pronav_setpoint", el.EntityId(1))
# plt.plot(ticks, df[:,0])
# plt.title('pronav command')

plt.show()

# Plot like this:
# python3 examples/rocket/plot.py