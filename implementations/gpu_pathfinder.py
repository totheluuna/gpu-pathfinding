from algorithms import a_star_v3
from utilities import helper
import numpy as np

import heapq

from numba import cuda, jit, njit, int32
import math
TPB = 16

# @jit
# def heuristic(x1, y1, x2, y2):
#     return np.int32(abs(x1-x2) + abs(y1-y2))

@njit
def heuristic(a, b):
    x1, y1 = a
    x2, y2 = b
    return abs(x1-x2) + abs(y1-y2)

@cuda.jit
def GPUPathfinder(grid, start, goal, hArray):
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid
    print(bpg)
    if x < grid.shape[0] and y < grid.shape[1]:
        goal_x, goal_y = goal
        if grid[x, y] != 0:
            # openList = pq.PriorityQueue()
            # hArray[x, y] = heuristic(x, y, goal_x, goal_y)
            hArray[x, y] = heuristic((x, y), goal)



    



