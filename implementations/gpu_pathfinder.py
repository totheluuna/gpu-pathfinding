from algorithms import a_star_v3
from utilities import helper
import numpy as np

from models import priority_queue_v2 as pq

from numba import cuda, jit, int32
import math
TPB = 16

@jit
def heuristic(x1, y1, x2, y2):
    return np.int32(abs(x1-x2) + abs(y1-y2))

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
            openList = pq.PriorityQueue()
            hArray[x, y] = heuristic(x, y, goal_x, goal_y)


