import configobj

configobj.scale_factor = 4 # scales to a power of 2
configobj.dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
configobj.TPB = 4
configobj.padded_TPB = TPB + 2
configobj.UNEXPLORED = int(math.pow(2, (scale_factor*2)))
# configobj.UNEXPLORED = 9999999
configobj.OPEN = 1
configobj.CLOSED = 0