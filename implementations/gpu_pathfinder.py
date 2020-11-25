from algorithms import a_star_v3
from utilities import helper
import numpy as np

from numba import cuda, jit, int32
import math
TPB = 16

@cuda.jit(int32(int32, int32))
def heuristic(a, b):
    x1, y1 = a
    x2, y2 = b
    return abs(x1-x2) + abs(y1-y2)

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
    print(bpg)
    if x < grid.shape[0] and y < grid.shape[1]:
        goal_x, goal_y = goal
        if grid[x, y] != 0:
            hArray[x, y] = heuristic(np.asarray((x,y), dtype=np.int32), goal)


