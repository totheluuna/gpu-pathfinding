import math
import numpy as np
from numba import njit
from timeit import default_timer as timer

import config
import helper
# functions for pathfinding
@njit
def passable(grid, tile):
    x,y = tile
    return grid[x,y] == 1
@njit
def inBounds(grid, tile):
    x, y = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
@njit
def getNeighbors(grid, tile, neighbors):
    # TO DO: modify to use numpy array
    (x, y) = tile
    for i in range(neighbors.size):
        if i == 0:
            neighbors[i,0] = x+1
            neighbors[i,1] = y
        elif i == 1:
            neighbors[i,0] = x+1
            neighbors[i,1] = y-1
        elif i == 2:
            neighbors[i,0] = x
            neighbors[i,1] = y-1
        elif i == 3:
            neighbors[i,0] = x-1
            neighbors[i,1] = y-1
        elif i == 4:
            neighbors[i,0] = x-1
            neighbors[i,1] = y
        elif i == 5:
            neighbors[i,0] = x-1
            neighbors[i,1] = y+1
        elif i == 6:
            neighbors[i,0] = x
            neighbors[i,1] = y+1
        elif i == 7:
            neighbors[i,0] = x+1
            neighbors[i,1] = y+1
@njit
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    # return abs(x1-x2) + abs(y1-y2)
    return int(math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2)))
@njit
def getMinIndex(arr):
    width, height = arr.shape
    min = arr[0,0]
    min_x = 0
    min_y = 0
    for i in range(width):
        for j in range(height):
            # print(arr[i,j])
            if arr[i,j] < min:
                min = arr[i,j]
                min_x = i
                min_y = j
    
    return min_x, min_y

@njit
def getMin(arr):
    width, height = arr.shape
    min = arr[0,0]
    for i in range(width):
        for j in range(height):
            if arr[i,j] < min:
                min = arr[i,j]
    
    return min

@njit
def search(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors):
    width, height = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal

    open[start_x, start_y] = 0
    g[start_x, start_y] = 0
    h[start_x, start_y] = heuristic(start, goal)
    cost[start_x, start_y] = g[start_x, start_y] + h[start_x, start_y]

    counter = 0
    # open_min = np.amin(open)
    open_min = getMin(open)
    while open_min < UNEXPLORED:
        current_x, current_y = getMinIndex(open)
        current = (current_x, current_y)
        # print(current, ' : ', open[current] )
        if current_x == goal_x and current_y == goal_y:
            break
        getNeighbors(grid, current, neighbors)
        for next in neighbors:
            if passable(grid, next) and inBounds(grid, next):
                next_x, next_y = next
                new_g = g[current_x, current_y] + 1
                if open[next_x, next_y] != UNEXPLORED:
                    if new_g < g[next_x, next_y]:
                        open[next_x, next_y] = UNEXPLORED
                if closed[next_x, next_y] != UNEXPLORED:
                    if new_g < g[next_x, next_y]:
                        closed[next_x, next_y] = UNEXPLORED
                if open[next_x, next_y] == UNEXPLORED and closed[next_x, next_y] == UNEXPLORED:
                    parents[next_x, next_y] = current_x * width + current_y
                    g[next_x, next_y] = new_g
                    h[next_x, next_y] = heuristic(next, goal)
                    cost[next_x, next_y] = g[next_x, next_y] + h[next_x, next_y]
                    open[next_x, next_y] = cost[next_x, next_y]
        closed[current_x, current_y] = cost[current_x, current_y]
        open[current_x, current_y] = UNEXPLORED
        counter += 1
        open_min = np.amin(open)

def test(grid, start, goal):
    # search for path
    print("----- Searching for Path -----")
    s = timer()
    width, height = grid.shape

    open = np.empty((width, height), dtype=np.int32)
    closed = np.empty((width, height), dtype=np.int32)
    parents = np.empty((width, height), dtype=np.int32)
    cost = np.empty((width, height), dtype=np.int32)
    g = np.empty((width, height), dtype=np.int32)
    h = np.empty((width, height), dtype=np.int32)
    neighbors = np.empty((8,2), dtype=np.int32)

    for i in range(width):
        for j in range(height):
            open[i,j] = config.UNEXPLORED
            closed[i,j] = config.UNEXPLORED
            parents[i,j] = -1
            cost[i,j] = 0
            g[i,j] = 0
            h[i,j] = 0
    for i in range(8):
        neighbors[i, 0] = 0
        neighbors[i, 1] = 0
        
    search(grid, start, goal, open, closed, parents, cost, g, h, config.UNEXPLORED, neighbors)
    x,y = start
    # path = []
    # reconstructPathV2(parents, tuple(start), tuple(goal), path)
    e = timer()
    # print(h)
    # print(parents)
    print('(Search + compilation) Path found in ', e-s, 's')

    time_ave = 0
    runs = 10
    for run in range(runs):
        s = timer()
        open = np.empty((width, height), dtype=np.int32)
        closed = np.empty((width, height), dtype=np.int32)
        parents = np.empty((width, height), dtype=np.int32)
        cost = np.empty((width, height), dtype=np.int32)
        g = np.empty((width, height), dtype=np.int32)
        h = np.empty((width, height), dtype=np.int32)
        neighbors = np.empty((8,2), dtype=np.int32)

        for i in range(width):
            for j in range(height):
                open[i,j] = config.UNEXPLORED
                closed[i,j] = config.UNEXPLORED
                parents[i,j] = -1
                cost[i,j] = 0
                g[i,j] = 0
                h[i,j] = 0
        for i in range(8):
            neighbors[i, 0] = 0
            neighbors[i, 1] = 0
        search(grid, start, goal, open, closed, parents, cost, g, h, config.UNEXPLORED, neighbors)
        e = timer()
        time_ave += (e-s)
        print('%dth search done in '%(run), e-s, 's')
    time_ave = time_ave/runs
    print('Average runtime in ', runs, ' runs: ', time_ave)
    # print(np.arange(config.dim[0]*config.dim[1]).reshape(config.dim).astype(np.int32))
    # print(parents)
    path = []
    helper.reconstructPathV2(parents, tuple(start), tuple(goal), path)
    print(path)

    return runs, time_ave, path