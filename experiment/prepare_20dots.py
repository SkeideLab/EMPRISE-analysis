import exp_utils
import pandas as pd
import os
import math
import numpy as np


"Script to prepare possible positions for the 20 dot stimulus. To be read in later during the experiment to reduce computation time"

cond = 'constarea'
shape = 'dot'
vf_radius = 0.75
num = 20

# total area that will be maintained over all conditions
# area = math.pi*radius**2
total_area = math.pi*0.2**2
radius_per_dot = np.sqrt((total_area / num) / math.pi)

sizes = [radius_per_dot]*num

dots = []

for i in range(5000):
    dots.append(exp_utils.get_positions('prep', vf_radius, [s*2 for s in sizes],
                shape, cond, max_search=500))

dots_df = pd.DataFrame(dots)
directory = os.path.dirname(os.path.abspath(__file__))
dots_path = os.path.join(directory, 'input_positions', '20dots.csv')
dots_df.to_csv(
    dots_path, header=False, index=False)

# a = pd.read_csv(
#     dots_path, header=None)
# one sample: list of points
# one points: list of two coordinates
# x = a.sample(axis=0, n=1)
# y = x.squeeze(axis=0).tolist()
# [[float(x.strip(' []')) for x in s.split(',')] for s in y]
