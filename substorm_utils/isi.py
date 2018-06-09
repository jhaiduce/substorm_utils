import numpy as np

def get_isi(onsets):
    onsets=np.sort(onsets)
    return onsets[1:]-onsets[:-1]
