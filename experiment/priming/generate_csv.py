import pandas as pd
import numpy as np
from itertools import product
import random
import os
def four_in_row(sequence):
    for i in range(len(sequence)-4):
        if len(np.unique(sequence[i:i+4])) == 1:
            return True
    return False
    

df = {}
modalities = ['dot', 'digit', 'word']
combos = list(product(modalities, modalities))
n_combs = 10
for start_run in range(n_combs):
    while True:
        random.shuffle(combos)
        # make a column for each start run
        # within each column there shall be a run order with tuples
        first_tup  = []
        for a,b in combos:
            first_tup.append(a)
        if four_in_row(first_tup):
           continue
        else:
            break
    df[f'col_{start_run}'] = combos

dir = os.path.dirname(os.path.abspath(__file__))
print(df)
df = pd.DataFrame(df)
df.to_csv(os.path.join(dir, 'run_order.csv'))