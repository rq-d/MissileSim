import argparse

import elodin as el
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import polars as pl
from sim import system, world
import random
import os

output_dir = "prompt/photos"

parser = argparse.ArgumentParser()
args = parser.parse_args()

nRuns = 50
simtimestep = 1.0 / 120.0
seed = random.seed(1)

# runs the sim on a given set of parameters
def runSim(takeoffPitch, targetX, timeSteps, Px, PyPz):

  exec = world(takeoffPitch, targetX, Px, PyPz).build(system())
  exec.run(timeSteps)

  return exec


# generate rand input variables
# run the sim
# store key params

inputs = {'runNum':[], 
          'takeoffPitchY': [], 
          'targetX': [], 
          'timeSteps': [],
          'Ix': [],
          'PyPz': []}
outputs = {'runNum':[],
           'downrangeDistance': [], 
           'maxAlpha': []}

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for i in range(nRuns):
  random.seed(i) # seed matches run number
  print("run ", i)
  # store inputs
  inputs['runNum'].append(          int(i))
  inputs['timeSteps'].append(       10000)
  inputs['takeoffPitchY'].append(   random.uniform(45, 80))
  inputs['targetX'].append(         random.uniform(-10000, -15000))
  inputs['Ix'].append(              0.1)
  inputs['PyPz'].append(            random.uniform(3,15))

  exec = runSim(inputs['takeoffPitchY'][-1], inputs['targetX'][-1], inputs['timeSteps'][-1], inputs['Ix'][-1], inputs['PyPz'][-1])

  # store outputs
  df = exec.history('world_pos', el.EntityId(1))
  df = df.with_columns(
      pl.col('world_pos').arr.get(4).alias('x'),
      pl.col('world_pos').arr.get(5).alias('y'),
      pl.col('world_pos').arr.get(6).alias('z'),
  )
  # downrange distance
  outputs['runNum'].append(int(i))
  d = np.linalg.norm(df.select(['x', 'y', 'z']).to_numpy()[-1])
  outputs['downrangeDistance'].append(float(np.linalg.norm(df.select(['x', 'y', 'z']).to_numpy()[-1])))
  
  # max alpha during the first n ticks (since bouncing off the ground adds noise to the signal)
  alpha = np.squeeze(exec.history("angle_of_attack", el.EntityId(1)).to_numpy())
  maxAlpha = np.max(alpha[:round(1/simtimestep*5000)])
  outputs['maxAlpha'].append(float(maxAlpha))

  ticks = np.arange(df.shape[0])
  ax1.plot(ticks,alpha, label='run '+str(i))
  ax1.set_ylim(-10, 10)
  ax1.set_xlim(0,10000)
  ax1.set_title("AoA")
  ax1.legend()

  # downrange distance plot
  distance = np.linalg.norm(df.select(["x", "y", "z"]).to_numpy(), axis=1)
  df = df.with_columns(pl.Series(distance).alias("distance"))
  ax2.plot(ticks, distance,label='run '+str(i))
  ax2.set_title("Downrange Distance (m)")
  ax2.legend()
  # miss distance (dont put this in for now)


plt.figure()
plt.scatter(inputs['takeoffPitchY'], outputs['downrangeDistance'], facecolors='none', edgecolors='black')
plt.title('Launch Angle vs Range')

plt.figure()
plt.scatter(inputs['takeoffPitchY'], outputs['maxAlpha'], facecolors='none', edgecolors='black')
plt.title('Launch Angle vs Pre-apogee Max AoA')

plt.figure()
plt.scatter(inputs['PyPz'], outputs['maxAlpha'], facecolors='none', edgecolors='black')
plt.title('Moment of Inertia Impact on max AoA')
plt.xlabel('Py, Pz')

# Save Figures and Dicts to file
print("Saving Figures and Sim files into ... " + output_dir )
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    filename = os.path.join(output_dir, f"figure_{fig_num}.png")
    fig.savefig(filename)
import json

with open(output_dir + "/simdata.json", "w") as f:
    json.dump([inputs, outputs], f, indent = 2)

plt.show()