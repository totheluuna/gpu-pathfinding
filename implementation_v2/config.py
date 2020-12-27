import math

seed = 4206969
scale_factor = 4 # scales to a power of 2
dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
TPB = 4
padded_TPB = TPB + 2
UNEXPLORED = int(math.pow(2, (scale_factor*2)))
# UNEXPLORED = 9999999
OPEN = 1
CLOSED = 0