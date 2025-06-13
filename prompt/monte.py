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

nRuns = 5
simtimestep = 1.0 / 120.0
seed = random.seed(1)

# runs the sim on a given set of parameters
def runSim(takeoffPitch, targetX, timeSteps):
  randpitch = random.randint(40,70)
  # exec = world(takeOffPitch=randpitch).build(system())
  exec = world(takeOffPitch=takeoffPitch, targetX=targetX).build(system())
  exec.run(timeSteps)
  return exec


# generate rand input variables
# run the sim
# store key params

inputs = {'runNum':[], 
          'takeoffPitchY': [], 
          'targetX': [], 
          'timeSteps': []}
outputs = {'runNum':[],
           'downrangeDistance': [], 
           'maxAlpha': []}

fig, ax = plt.subplots()
# Run the sim n times
for i in range(nRuns):
  random.seed(i) # seed matches run number

  # store inputs
  inputs['runNum'].append(          int(i))
  inputs['takeoffPitchY'].append(   random.uniform(45, 80))
  inputs['targetX'].append(         random.uniform(-10000, -15000))
  inputs['timeSteps'].append(       10000)

  exec = runSim(inputs['takeoffPitchY'][-1], inputs['targetX'][-1], inputs['timeSteps'][-1])

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
  ax.plot(ticks,alpha)
  ax.set_ylim(-5, 5)
  ax.set_xlim(0,2000)
  ax.set_title("AoA(t)")

  # miss distance (dont put this in for now)


plt.figure()
plt.scatter(inputs['takeoffPitchY'], outputs['downrangeDistance'])
plt.title('Range vs Launch Angle')

plt.figure()
plt.scatter(inputs['takeoffPitchY'], outputs['maxAlpha'])
plt.title('Max AoA vs Launch Angle')

plt.show()
print(inputs, '\n')
print(outputs, '\n')