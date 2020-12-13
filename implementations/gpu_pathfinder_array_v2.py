from random import randint, seed

import cv2 as cv
import sys
import os
import math
import numpy as np
import cupy as cp

import heapq
from timeit import default_timer as timer

from numba import cuda, int32, typeof

OPEN = 1
CLOSED = 0

scale_factor = 4 # scales to a power of 2
dim = (int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor)))
UNEXPLORED = int(math.pow(2, (scale_factor*2)))
TPB = 4


seed(42069)
# functions for converting images to grids
def getListOfFiles(dirName, allFiles):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
def imageToGrid(image, grid, dim):
    print("----- Converting Image to Grid Object -----")
    # load image
    img = cv.imread(cv.samples.findFile(image))
    if img is None:
        sys.exit("Could not read the image.")
    cv.imwrite("output/original.png", img)

    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    cv.imwrite("output/resized.png", resized) 
    print('Resized Dimensions : ',resized.shape) 

    # convert to grayscale
    grayImg = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    cv.imwrite("output/grayscale.png", grayImg)
    
    # apply thresholding to easily separate walkable and unwalkable areas
    ret, thresh1 = cv.threshold(grayImg,75,229,cv.THRESH_BINARY)
    cv.imwrite("output/threshold.png", thresh1)
    print('Threshold Dimensions : ', thresh1.shape)

    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            # if the current value = 0 (meaning black) append to list of walls
            if thresh1[i][j] == 0:
                grid[i,j] = 0
            else:
                grid[i,j] = 1

def createGridFromDatasetImage(dataset, grid, dim):
    print('----- Creating Grid Object from Dataset Image-----')
    listOfImages = []
    getListOfFiles(dataset, listOfImages)
    image = listOfImages[randint(0, len(listOfImages)-1)]
    print('Random Image: ', image)
    imageToGrid(image, grid, dim)

def randomStartGoal(grid, start, goal):
    print('----- Generating Random Start and Goal -----')
    dist = 0
    width, height = grid.shape
    hypotenuse = math.sqrt(math.pow(width, 2) + math.pow(height,2))
    while dist <= 0.50*hypotenuse:
        random_x = randint(0, width-1)
        random_y = randint(0, height-1)
        start[0] = random_x
        start[1] = random_y
        while grid[random_x, random_y] == 0:
            random_x = randint(0, width-1)
            random_y = randint(0, height-1)
            start[0] = random_x
            start[1] = random_y

        random_x = randint(0, width-1)
        random_y = randint(0, height-1)
        goal[0] = random_x
        goal[1] = random_y
        while grid[random_x, random_y] == 0:
            random_x = randint(0, width-1)
            random_y = randint(0, height-1)
            goal[0] = random_x
            goal[1] = random_y
        a = np.array(start)
        b = np.array(goal)
        dist = np.linalg.norm(a-b)


# function for reconstructing found path
def reconstructPathV2(cameFrom, start, goal, path):
    currentX, currentY = goal
    while (currentX, currentY) != start:
        path.append((currentX, currentY))
        currentX, currentY = cameFrom[currentX, currentY]
    path.append(start)
    path.reverse

# functions for pathfinding
@cuda.jit(device=True)
def passable(grid, tile):
    x,y = tile
    return grid[x,y] == int32(1)
@cuda.jit(device=True)
def inBounds(grid, tile):
    x, y = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
@cuda.jit(device=True)
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
@cuda.jit(device=True)
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)
@cuda.jit(device=True)
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

@cuda.jit(device=True)
def getMin(arr):
    width, height = arr.shape
    min = arr[0,0]
    for i in range(width):
        for j in range(height):
            if arr[i,j] < min:
                min = arr[i,j]
    
    return min


@cuda.jit(device=True)
def search(grid, start, goal, open, closed, parents, cost, g, h, neighbors, block):
# def search(x, y, grid, start, goal, open, closed, parents, cost, g, h, neighbors):
    width, height = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal

    open[start_x, start_y] = 0
    g[start_x, start_y] = 0
    # h[start_x, start_y] = heuristic(start, goal)
    cost[start_x, start_y] = g[start_x, start_y] + h[start_x, start_y]

    counter = 0
    # while np.amin(open) < UNEXPLORED:
    while getMin(open) < UNEXPLORED:
        current_x, current_y = getMinIndex(open)
        current = (current_x, current_y)
        if (current_x == goal_x and current_y == goal_y):
        # or (block[current_x, current_y] != block[start_x, start_y]):
            break
        getNeighbors(grid, current, neighbors)
        for next in neighbors:
            if inBounds(grid, next):
                if passable(grid, next):
                    next_x, next_y = next
                    new_g = g[current_x, current_y] + 1
                    if open[next_x, next_y] != UNEXPLORED:
                        if new_g < g[next_x, next_y]:
                            open[next_x, next_y] = UNEXPLORED
                    if closed[next_x, next_y] != UNEXPLORED:
                        if new_g < g[next_x, next_y]:
                            closed[next_x, next_y] = UNEXPLORED
                    if open[next_x, next_y] == UNEXPLORED and closed[next_x, next_y] == UNEXPLORED:
                        # parents[next_x, next_y, 0] = current_x
                        # parents[next_x, next_y, 1] = current_y
                        parents[next_x, next_y] = current_x * TPB + current_y
                        # parents[next_x, next_y] = current_x * width + current_y
                        g[next_x, next_y] = new_g
                        # h[next_x, next_y] = heuristic(next, goal) # omit this step since H is precomputed on GPU
                        cost[next_x, next_y] = g[next_x, next_y] + h[next_x, next_y]
                        open[next_x, next_y] = cost[next_x, next_y]
        closed[current_x, current_y] = cost[current_x, current_y]
        open[current_x, current_y] = UNEXPLORED
        counter += 1

@cuda.jit
def GridDecompPathV2(grid, planning_grid, start, goal, parents, h, block):
    x, y = cuda.grid(2)
    glb_x, glb_y = dim
    goal_x, goal_y = goal

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= grid.shape[0] and y >= grid.shape[1]:
        return

    # print('running thread: ', tx, ty)
    # print('grid coordinates: ', x, y)
    if passable(grid, (x,y)) and (x != goal_x or y != goal_y):
        # initialize local arrays
        local_open = cuda.local.array((TPB, TPB), int32)
        local_closed = cuda.local.array((TPB, TPB), int32)
        local_cost = cuda.local.array((TPB, TPB), int32)
        local_g = cuda.local.array((TPB, TPB), int32)
        local_neighbors = cuda.local.array((8,2), int32)

        for i in range(TPB):
            for j in range(TPB):
                local_open[i,j] = UNEXPLORED
                local_closed[i,j] = UNEXPLORED
                local_cost[i,j] = 0
                local_g[i,j] = 0
        # cuda.syncthreads()

        for i in range(8):
            local_neighbors[i, 0] = 0
            local_neighbors[i, 1] = 0
# (goal[0]%TPB, goal[1]%TPB)
        # cuda.syncthreads()
        search(x, y, planning_grid[block[x,y]], (tx, ty), goal , local_open, local_closed, parents[x,y], local_cost, local_g, h[block[x,y]], local_neighbors, block)

@cuda.jit
def computeHeuristics(grid, start, goal, h_start, h_goal, block):
    x, y = cuda.grid(2)
    width, height = grid.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    dim_x = cuda.blockDim.x
    dim_y = cuda.blockDim.y
    bpg = cuda.gridDim.x    # blocks per grid
    if x < grid.shape[0] and y < grid.shape[1]:
        if passable(grid, (x,y)) and inBounds(grid, (x,y)):
            h_goal[x,y] = heuristic((x,y), goal)
            h_start[x,y] = heuristic((x,y), start)
        block[x,y] = bx * dim_x + by
        cuda.syncthreads()

@cuda.jit
def SimultaneousLocalSearch(blocked_grid, local_start, local_goal, blocked_h_goal, blocked_h_start, local_parents, block):
    i = cuda.grid(1)
    if i >= blocked_grid.shape[0]:
        return
    
    if passable(blocked_grid[i], local_start[i]) and inBounds(blocked_grid[i], local_start[i]):
        print('locally searching path in block %d' %(i))
    
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def main():
    print('----- Preparing Grid -----')
    # create grid from image dataset
    grid = np.zeros(dim, dtype=np.int32)
    # grid = np.ones(dim, dtype=np.int32)
    createGridFromDatasetImage('dataset/da2-png', grid, dim)
    print(grid)

    # generate random start and goal
    start = [-1, -1]
    goal = [-1, -1]
    # start = [0, 0]
    # goal = [grid.shape[0]-1, grid.shape[1]-1]
    randomStartGoal(grid, start, goal)
    start = np.array(start)
    goal = np.array(goal)
    print(start)
    print(goal)

    # debugging purposes: use guide for 1D mapping of indexes
    guide = np.arange(dim[0]*dim[1]).reshape(dim).astype(np.int32)
    x, y = start
    guide[x,y] = 696
    x, y = goal
    guide[x,y] = 696

    # initialize essential arrays for search algorithm
    print('----- Initializing Variables -----')

    H_goal = np.empty(dim, dtype=np.int32)
    H_goal[:] = UNEXPLORED
    H_start = np.empty(dim, dtype=np.int32)
    H_start[:] = UNEXPLORED
    block = np.zeros(dim, dtype=np.int32)

    # compute heuristics towards start and goal
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print('----- Computing Heuristics -----')
    computeHeuristics[blockspergrid, threadsperblock](grid, start, goal, H_start, H_goal, block)
    print('Start H: ')
    print(H_start)
    print('Goal H: ')
    print(H_goal)
    print('Blocking: ')
    print(block)
    print('Index Guide: ')
    print(guide)

    # reshape grid, H_start, H_goal into separate blocks
    blocked_grid = blockshaped(grid, TPB, TPB)
    blocked_H_start = blockshaped(H_start, TPB, TPB)
    blocked_H_goal = blockshaped(H_goal, TPB, TPB)

    start_block = block[start[0], start[1]]
    goal_block = block[goal[0], goal[1]]
    print('BLOCKS INCLUDING START AND GOAL: ')
    print('start block: ', start_block)
    print(blocked_grid[start_block])
    print('goal block ', goal_block)
    print(blocked_grid[goal_block])

    print('GRID BLOCKS: ')
    print(blocked_grid)
    print()
    print('H_start BLOCKS: ')
    print(blocked_H_start)
    print()
    print('H_goal BLOCKS: ')
    print(blocked_H_goal)

    # determine local starts and local goals for all blocks
    local_start = np.zeros((blocked_grid.shape[0], 2), np.int32)
    local_goal = np.zeros((blocked_grid.shape[0], 2), np.int32)
    for i in range(blocked_grid.shape[0]):
        # find the (x,y) index of the min value in each H_start and H_goal block
        local_start[i] = np.array(np.unravel_index(blocked_H_goal[i].argmin(), blocked_H_goal[i].shape))
        local_goal[i] = np.array(np.unravel_index(blocked_H_start[i].argmin(), blocked_H_start[i].shape))
        print('-- %dth block --' %(i))
        print('local goal: ', local_goal[i])
        print('local start: ', local_start[i])
    
    # parents array contains info where tiles came from
    local_parents = np.empty(blocked_grid.shape, np.int32)

    # Simultaneous local search
    SimultaneousLocalSearch[blockspergrid, threadsperblock](blocked_grid, local_start, local_goal, blocked_H_goal, blocked_H_start, local_parents, block))

    print('Something')


    


    

    

    



    # width, height = grid.shape
    # # parents = cp.empty((width, height), dtype=cp.int32)
    # parents = cp.empty((TPB, TPB), dtype=cp.int32)
    # parents[:] = -1

    # # h = cp.zeros((width, height), dtype=cp.int32)
    # h = cp.empty((width, height), dtype=cp.int32)
    # h[:] = UNEXPLORED
    # block = cp.zeros((width, height), dtype=cp.int32)

    # # parents_arr = cp.empty((width, height, width, height), dtype=cp.int32)
    # parents_arr = cp.empty((width, height, TPB, TPB), dtype=cp.int32)
    # parents_arr[:] = parents

    # print('PARENTS ARRAY: ')
    # print(parents_arr)

    # path = []
    # threadsperblock = (TPB, TPB)
    # blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
    # blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # print('----- Precomputing Heuristics -----')
    # precomputeHeuristics[blockspergrid, threadsperblock](grid, start, goal, h, block)
    # print("GUIDE:")
    # print(guide)
    # print("H:")
    # print(h)
    # print("BLOCKING:")
    # print(block)

    # # threadsperblock = (TPB, TPB)
    # # blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
    # # blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
    # # blockspergrid = (blockspergrid_x, blockspergrid_y)

    # start_block = block[start[0], start[1]]
    # goal_block = block[goal[0], goal[1]]
    # print('BLOCKS INCLUDING START AND GOAL: ')
    # print('start block: ', start_block)
    # print(planning_grid[start_block])
    # print('goal block ', goal_block)
    # print(planning_grid[goal_block])

    # planning_h = blockshaped(h, TPB, TPB)
    # print('RESHAPED H: ')
    # for i in range(planning_h.shape[0]):
    #     print(' BLOCK %d: '%(i))
    #     print(planning_h[i])
    #     print()

    # # print("----- Searching for Path -----")
    # # s = timer()
    # # # GridDecompPath[blockspergrid, threadsperblock](grid, start, goal, parents_arr, h, block)
    # # # local_goal = np.array([goal[0]%TPB, goal[1]%TPB])
    # # # print('LOCAL GOAL: ', local_goal)
    # # GridDecompPathV2[blockspergrid, threadsperblock](grid, planning_grid, start, goal, parents_arr, planning_h, block)
    # # for i in range(parents_arr.shape[0]):
    # #     for j in range(parents_arr.shape[1]):
    # #         print('tile: ', (i,j))
    # #         print(parents_arr[i, j])
    # #         print()
    # # # path = []
    # # # reconstructPathV2(parents_arr[x,y], tuple(start), tuple(goal), path)
    # # # print(path)
    # # # print(parents_arr)
    # # # parents_host = parents_arr.get()
    # # print(parents_arr[start[0], start[1]])
    # # e = timer()
    # # print('Kernel Launch done in ', e-s, 's')

    # # time_ave = 0
    # # runs = 10
    # # for i in range(runs):
    # #     s = timer()
    # #     GridDecompPathV2[blockspergrid, threadsperblock](grid, planning_grid, start, goal, parents_arr, planning_h, block)
    # #     parents_host = parents_arr.get()
    # #     # print(block)
    # #     # TODO: reconstruct path
    # #     e = timer()
    # #     time_ave += (e-s)
    # #     print('%dth kernel Launch done in ' %(i), e-s, 's')
    # # time_ave = time_ave/runs
    # # print('Average runtime in ', runs, ' runs: ', time_ave)
    

if __name__ == "__main__":
    main()

    


