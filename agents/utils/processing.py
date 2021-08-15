from collections import OrderedDict

import numpy as np

def stateify(observation: OrderedDict, keys = None):
    output = np.array([])
    keys = keys or observation.keys()
    for key in keys:
        output = np.append(output, observation[key])
    return output
