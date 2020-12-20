import sys,os
o_path = os.getcwd()
sys.path.append(o_path)
import utils.modern_robotics as rob 
import numpy as np 

print(rob.NearZero(np.array([1,2])))