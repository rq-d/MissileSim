import parsemotordata
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
thrust_curve = parsemotordata.parse(current_dir+"/data/AeroTech_M685W.txt")

print(thrust_curve)