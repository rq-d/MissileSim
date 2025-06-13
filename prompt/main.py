from sim import SIM_TIME_STEP, system, world

# world().run(system(), sim_time_step=SIM_TIME_STEP, max_ticks=1200)
# SIM_TIME_STEP = 1.0 / 120.0
world().run(system(), sim_time_step=SIM_TIME_STEP, max_ticks=10000)