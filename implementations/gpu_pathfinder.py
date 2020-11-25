from algorithms import a_star_v3
from utilities import helper
import numpy as np

from numba import cuda

TPB = 16

@cuda.jit
def GPUPathfinder(grid, start, goal, hArray):
    # path = []
    # parents = {}
    # FCost = {}
    # if algorithm == 'a_star':
    #     parents, FCost = a_star_v3.search(graph, start, goal)
    #     path = helper.reconstructPathV2(parents, start, goal)
    # else:
    #     print("No implementation of the search algorithm")
    # return FCost, path

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x < grid.shape[0] and y < grid.shape[1]:
        hArray[x, y] *= 10


