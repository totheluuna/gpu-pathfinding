from random import randint, seed
import argparse
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

scale_factor = 5 # scales to a power of 2
dim = (int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor)))
UNEXPLORED = int(math.pow(2, (scale_factor*2)))
TPB = 4

# seed(42042042069)
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
def reconstructPathV2(parents, start, goal, path):
    width, height = dim
    # currentX, currentY = goal
    # while (currentX, currentY) != start:
    #     path.append((currentX, currentY))
    #     currentX, currentY = parents[currentX, currentY]
    # path.append(start)
    # path.reverse
    start_x, start_y = start
    start_1d_index = start_x * width + start_y
    current_x, current_y = goal
    current_1d_index = current_x * width + current_y
    print('START: (%d, %d) -> %d' %(start_x, start_y, start_1d_index))
    # print('CURRENT (GOAL): (%d, %d) -> %d' %(current_x, current_y, current_1d_index))
    
    while current_1d_index != start_1d_index:
        print('CURRENT (GOAL): (%d, %d) -> %d' %(current_x, current_y, current_1d_index))
        path.append(current_1d_index)
        parent_1d_index = parents[current_x, current_y]
        current_x = int((parent_1d_index-(parent_1d_index%width))/width)
        current_y = parent_1d_index%width 
        current_1d_index = current_x * width + current_y
    path.append(start_1d_index)
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
                        # parents[next_x, next_y] = current_x * TPB + current_y
                        parents[next_x, next_y] = current_x * width + current_y
                        g[next_x, next_y] = new_g
                        # h[next_x, next_y] = heuristic(next, goal) # omit this step since H is precomputed on GPU
                        cost[next_x, next_y] = g[next_x, next_y] + h[next_x, next_y]
                        open[next_x, next_y] = cost[next_x, next_y]
        closed[current_x, current_y] = cost[current_x, current_y]
        open[current_x, current_y] = UNEXPLORED
        counter += 1

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
    # pos = cuda.grid(1)
    # if pos >= blocked_grid.shape[0]:
    #     return
    x, y = cuda.grid(2)
    bpg = cuda.gridDim.x    # blocks per grid
    pos = x * bpg + y
    if pos >= blocked_grid.shape[0]:
        return 
    
    # if passable(blocked_grid[pos], local_start[pos]) and inBounds(blocked_grid[pos], local_start[pos]):
    # initialize essential local arrays
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
    cuda.syncthreads()
    for i in range(8):
        local_neighbors[i, 0] = 0
        local_neighbors[i, 1] = 0
    cuda.syncthreads()

    search(blocked_grid[pos], local_start[pos], local_goal[pos], local_open, local_closed, local_parents[pos], local_cost, local_g, blocked_h_goal[pos], local_neighbors, block)
    cuda.syncthreads()
@cuda.jit
def MapBlocks(guide, parents):
    x, y = cuda.grid(2)
    width, height = guide.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    dim_x = cuda.blockDim.x
    dim_y = cuda.blockDim.y

    if x >= width and y >= height:
        return

    if parents[x,y] > -1:
        if parents[x,y] != guide[x,y]:
            index = parents[x,y]
            _x = int((index-(index%width))/width)
            _y = index%width
            parents[x,y] = guide[_x, _y]
    # cuda.syncthreads()
@cuda.jit
def MapBlocks2(guide, parents, h):
    x, y = cuda.grid(2)
    width, height = guide.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    dim_x = cuda.blockDim.x
    dim_y = cuda.blockDim.y

    if x >= width and y >= height:
        return

    if parents[x,y] == guide[x,y]:
        # initialize local array
        local_neighbors = cuda.local.array((8,2), int32)
        for i in range(8):
            local_neighbors[i, 0] = 0
            local_neighbors[i, 1] = 0
        # get neighbors of (x,y)
        getNeighbors(parents, (x,y), local_neighbors)
        min_x, min_y = (x, y) # tile with the minimum heuristic distance from the start tile
        for i in range(8):
            _x = local_neighbors[i, 0]
            _y = local_neighbors[i, 1]
            if h[_x, _y] < h[min_x, min_y]:
                min_x, min_y = (_x, _y)

        parents[x,y] = min_x * width + min_y
        # cuda.syncthreads()

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

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def main():
    # global scale_factor
    # global TPB

    # parser = argparse.ArgumentParser(description='GPU Pathfinding')
    # parser.add_argument('scale_factor', type=int, help='Scale factor (power of 2)')
    # parser.add_argument('TPB', type=int, help='Block width')
    # args = parser.parse_args()
    # scale_factor = args.scale_factor
    # TPB = args.TPB

    print('----- Preparing Grid -----')
    # create grid from image dataset
    # grid = np.zeros(dim, dtype=np.int32)
    grid = np.ones(dim, dtype=np.int32)
    # createGridFromDatasetImage('dataset/da2-png', grid, dim)
    print(grid)

    # generate random start and goal
    # start = [-1, -1]
    # goal = [-1, -1]
    start = [0, 0]
    goal = [grid.shape[0]-1, grid.shape[1]-1]
    # randomStartGoal(grid, start, goal)
    start = np.array(start)
    goal = np.array(goal)
    print(start)
    print(goal)

    # debugging purposes: use guide for 1D mapping of indexes
    guide = np.arange(dim[0]*dim[1]).reshape(dim).astype(np.int32)
    # x, y = start
    # guide[x,y] = 696
    # x, y = goal
    # guide[x,y] = 696

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
    print('THREADS PER BLOCK: ',threadsperblock)
    print('BLOCKS PER GRID: ', blockspergrid)
    print('----- Computing Heuristics -----')
    computeHeuristics[blockspergrid, threadsperblock](grid, start, goal, H_start, H_goal, block)
    # print('Start H: ')
    # print(H_start)
    # print('Goal H: ')
    # print(H_goal)
    # print('Blocking: ')
    # print(block)
    # print('Index Guide: ')
    # print(guide)

    # reshape grid, H_start, H_goal into separate blocks
    print('----- Reshaping/Decomposing grid and heuristics to separate blocks -----')
    blocked_grid = blockshaped(grid, TPB, TPB)
    blocked_H_start = blockshaped(H_start, TPB, TPB)
    blocked_H_goal = blockshaped(H_goal, TPB, TPB)
    blocked_guide = blockshaped(guide, TPB, TPB)
    blocked_block = blockshaped(block, TPB, TPB)

    start_block = block[start[0], start[1]]
    goal_block = block[goal[0], goal[1]]\

    # debug stuff
    # print(blocked_H_goal.shape)
    # for i in range(blocked_H_goal.shape[0]):
    #     print('%dth block: '%(i))
    #     print(blocked_H_goal[i])
    #     print()

    # print('BLOCKS INCLUDING START AND GOAL: ')
    # print('start block: ', start_block)
    # print(blocked_guide[start_block])
    # print('goal block: ', goal_block)
    # print(blocked_guide[goal_block])
    # print('BLOCKED SHAPE: ' , blocked_guide.shape)

    # # print('GRID BLOCKS: ')
    # # print(blocked_grid)
    # # print()
    # # print('H_start BLOCKS: ')
    # # print(blocked_H_start)
    # # print()
    # # print('H_goal BLOCKS: ')
    # # print(blocked_H_goal)

    # parents array contains info where tiles came from
    local_parents = np.empty(blocked_grid.shape, np.int32)
    local_parents[:] = -1

    # determine local starts and local goals for all blocks
    print('----- Determining local starts and goals for all blocks -----')
    local_start = np.zeros((blocked_grid.shape[0], 2), np.int32)
    local_goal = np.zeros((blocked_grid.shape[0], 2), np.int32)
    for i in range(blocked_grid.shape[0]):
        # find the (x,y) index of the min value in each H_start and H_goal block
        local_goal[i] = np.array(np.unravel_index(blocked_H_goal[i].argmin(), blocked_H_goal[i].shape))
        local_start[i] = np.array(np.unravel_index(blocked_H_start[i].argmin(), blocked_H_start[i].shape))
        x, y = local_start[i]
        local_parents[i, x, y] = blocked_guide[i, x, y]
        x, y = local_goal[i]
        local_parents[i, x, y] = blocked_guide[i, x, y]

        # print('-- %dth block --' %(i))
        # print('local goal: ', local_goal[i])
        # print('local start: ', local_start[i])

    # print(local_parents)
    # parents = unblockshaped(local_parents, dim[0], dim[1])
    # print(guide)
    # print(parents)

    # debug stuff
    # print(local_parents.shape)
    # for i in range(local_parents.shape[0]):
    #     print('%dth block: '%(i))
    #     print(local_start[i])
    #     print(local_goal[i])
    #     print()

    # Simultaneous local search
    s = timer()
    SimultaneousLocalSearch[blockspergrid, threadsperblock](blocked_grid, local_start, local_goal, blocked_H_goal, blocked_H_start, local_parents, block)
    print(local_parents)
    e = timer()
    print('kernel launch (+ compilation) done in ', e-s, 's')

    # # debug stuff
    # print(local_parents.shape)
    # for i in range(local_parents.shape[0]):
    #     print('%dth block: '%(i))
    #     print(local_parents[i])
    #     print()

    time_ave = 0
    runs = 10
    for run in range(runs):
        s = timer()
        SimultaneousLocalSearch[blockspergrid, threadsperblock](blocked_grid, local_start, local_goal, blocked_H_goal, blocked_H_start, local_parents, block)
        print(local_parents[local_parents.shape[0]-1])
        # print(block)
        e = timer()
        time_ave += (e-s)
        print('%dth kernel launch done in ' %(run), e-s, 's')
    time_ave = time_ave/runs
    print('Average runtime in ', runs, ' runs: ', time_ave)

    # print(local_parents.shape)
    # for i in range(local_parents.shape[0]):
    #     print('%dth block: '%(i))
    #     print(local_parents[i])
    #     print()

    
    # # TODO: reconstruct path
    print(local_parents.shape)
    blocked_guide_gpu = cp.array(blocked_guide)
    local_parents_gpu = cp.array(local_parents)
    for i in range(local_parents.shape[0]):
        # print(i)
        MapBlocks[blockspergrid, threadsperblock](blocked_guide[i], local_parents[i])
        # MapBlocks[blockspergrid, threadsperblock](blocked_guide_gpu[i], local_parents_gpu[i])

    print(local_parents.shape)
    for i in range(local_parents.shape[0]):
        print('%dth block: '%(i))
        print(local_parents[i])
        print()
    

    parents = unblockshaped(local_parents, dim[0], dim[1])
    print(parents)

    # # neighbors = cp.zeros((dim[0], dim[1], 8, 2), cp.int32)
    MapBlocks2[blockspergrid, threadsperblock](guide, parents, H_start)
    # print(guide)
    print(parents)
    path = []
    reconstructPathV2(parents, start, goal, path)
    print(path)

if __name__ == "__main__":
    main()

    


