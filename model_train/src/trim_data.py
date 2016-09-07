import os
import sys
import numpy as np

for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
    for f in [ff for ff in filenames if ff.endswith(".npy")]:
        x = np.load(f)
        # values encoded with a 3 or 4 need to be removed
        badvals = [3, 4]
        ix = np.in1d(x.ravel(), badvals)
        index = np.where(ix)

