"""
simulated data
"""
import numpy as np
from hmm import HMM

def get_input_data(N, T, V):
    return np.random.randint(0, V, size = (N, T))

def get_state(H):
    return np.array(range(0, H))


# simulation
V = [0, 1, 2, 3, 4]
X = get_input_data(20, 10, len(V))
Y = get_state(3)

# training
model = HMM()
model.fit(X, Y, V)

t = model.Transition
e = model.Emission
p = model.Pi

# decoding
model = HMM(t, e, p)
res = model.decode(np.random.randint(0, len(V), size = (2, 10)))
print(res)