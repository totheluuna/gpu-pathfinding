import config

config.scale_factor = 4 # scales to a power of 2
config.dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
config.TPB = 4
config.padded_TPB = TPB + 2
config.UNEXPLORED = int(math.pow(2, (scale_factor*2)))
# config.UNEXPLORED = 9999999
config.OPEN = 1
config.CLOSED = 0