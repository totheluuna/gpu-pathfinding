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
from skimage.util.shape import view_as_windows

scale_factor = 3 # scales to a power of 2
dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
TPB = 4
padded_TPB = TPB + 2

# UNEXPLORED = int(math.pow(2, (scale_factor*2)))
UNEXPLORED = 9999999
OPEN = 1
CLOSED = 0


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
    print(image)
    # image = 'dataset/da2-png/ht_chantry.png'
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
    # print('START: (%d, %d) -> %d' %(start_x, start_y, start_1d_index))
    # print('CURRENT (GOAL): (%d, %d) -> %d' %(current_x, current_y, current_1d_index))
    
    while current_1d_index != start_1d_index:
        # print('CURRENT (GOAL): (%d, %d) -> %d' %(current_x, current_y, current_1d_index))
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

@cuda.jit(device=True)
def searchV2(x, y, grid, start, goal, open, closed, parents, cost, g, h, neighbors, block, guide, counter_arr):
    width, height = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal
    goal_1d_index = goal_x * dim[0] + goal_y

    open[start_x, start_y] = 0
    g[start_x, start_y] = 0
    cost[start_x, start_y] = g[start_x, start_y] + h[start_x, start_y]
    # parents[start_x, start_y] = start_x*width+start_y
    parents[start_x, start_y] = 729 

    counter = 0
    _min = getMin(open)
    while _min < UNEXPLORED:
        current_x, current_y = getMinIndex(open)
        current = (current_x, current_y)
        actual_index = guide[current]
        # print(start_x, start_y, _min, current_x, current_y, block[current], actual_index)
        if (actual_index == goal_1d_index) or (block[start] != block[current]):
            # print("\riterations: {}".format(counter), end='')
            counter_arr[x,y] = counter
            break
        getNeighbors(grid, current, neighbors)
        for next in neighbors:
            if passable(grid, next) and inBounds(grid, next):
                row, col = next
                idx = row*width+col
                # print(idx, 'is passable and in bounds')
                next_x, next_y = next
                new_g = g[current_x, current_y] + 1
                if open[next_x, next_y] != UNEXPLORED:
                    if new_g < g[next_x, next_y]:
                        open[next_x, next_y] = UNEXPLORED
                if closed[next_x, next_y] != UNEXPLORED:
                    if new_g < g[next_x, next_y]:
                        closed[next_x, next_y] = UNEXPLORED
                if open[next_x, next_y] == UNEXPLORED and closed[next_x, next_y] == UNEXPLORED:
                    # parents[next_x, next_y] = np.array([current_x, current_y])
                    # parents[next_x, next_y] = current_x * width + current_y
                    parents[next_x, next_y] = actual_index 
                    g[next_x, next_y] = new_g
                    cost[next_x, next_y] = g[next_x, next_y] + h[next_x, next_y]
                    open[next_x, next_y] = cost[next_x, next_y]
        closed[current_x, current_y] = cost[current_x, current_y]
        open[current_x, current_y] = UNEXPLORED
        counter = counter + 1

@cuda.jit
def computeHeuristics(grid, start, goal, h_start, h_goal):
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
        cuda.syncthreads()

@cuda.jit
# def GridDecompSearch(grid, start, goal, h, block, parents, grid_blocks, guide_blocks, h_blocks, blocks):
def GridDecompSearch(grid, h, block, grid_blocks, start, goal, parents, h_blocks, guide_blocks, blocks, counter):
    x, y = cuda.grid(2)
    width, height = dim
    bpg = cuda.gridDim.x    # blocks per grid
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= width and y >= height:
        return 
    
    local_bound_check = tx == 0 or tx == TPB-1 or ty == 0 or ty == TPB-1
    start_tile_check = x == start[0] and y == start[1]
    goal_tile_check = x == goal[0] and y == goal[1]
    thread_block = block[x,y]

    if local_bound_check or start_tile_check or goal_tile_check:
        # print(thread_block)
        # counter[x,y] = 1
        # initialize essential local arrays
        local_grid = grid_blocks[thread_block]
        local_block = blocks[thread_block]
        local_guide = guide_blocks[thread_block]
        local_start = (tx+1, ty+1)
        local_h = h_blocks[thread_block]
        local_ctr = 0

        _open = cuda.local.array((padded_TPB, padded_TPB), int32)
        _closed = cuda.local.array((padded_TPB, padded_TPB), int32)
        _cost = cuda.local.array((padded_TPB, padded_TPB), int32)
        _g = cuda.local.array((padded_TPB, padded_TPB), int32)
        _neighbors = cuda.local.array((8,2), int32)
        

        for i in range(padded_TPB):
            for j in range(padded_TPB):
                _open[i,j] = UNEXPLORED
                _closed[i,j] = UNEXPLORED
                _cost[i,j] = 0
                _g[i,j] = 0
    
        for i in range(8):
            _neighbors[i, 0] = 0
            _neighbors[i, 1] = 0
        
        searchV2(x, y, grid_blocks[thread_block], (tx+1, ty+1), goal, _open, _closed, parents[x,y], _cost, _g, h_blocks[thread_block], _neighbors, blocks[thread_block], guide_blocks[thread_block], counter)
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
            cuda.syncthreads()
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

@cuda.jit
def padGrid(grid, padded_grid):
    x, y = cuda.grid(2)
    width, height = dim

    if x >= width and y >= height:
        return
    padded_grid[x+1, y+1] = grid[x, y]
    cuda.syncthreads()

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
    # global dim

    # parser = argparse.ArgumentParser(description='GPU Pathfinding')
    # parser.add_argument('scale_factor', type=int, help='Scale factor (power of 2)')
    # parser.add_argument('TPB', type=int, help='Block width')
    # args = parser.parse_args()
    # scale_factor = args.scale_factor
    # TPB = args.TPB
    # dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
    
    width, height = dim

    print('----- Preparing Grid -----')
    # create grid from image dataset
    # grid = np.zeros(dim, dtype=np.int32)
    # createGridFromDatasetImage('dataset/da2-png', grid, dim)
    grid = np.ones(dim, dtype=np.int32)

    # generate random start and goal
    # start = [-1, -1]
    # goal = [-1, -1]
    # randomStartGoal(grid, start, goal)
    start = [0, 0]
    goal = [grid.shape[0]-1, grid.shape[1]-1]
    start = np.array(start)
    goal = np.array(goal)
    

    # debugging purposes: use guide for 1D mapping of indexes
    guide = np.arange(dim[0]*dim[1]).reshape(dim).astype(np.int32)

    # initialize essential arrays for search algorithm
    print('----- Initializing Variables -----')

    H_goal = np.empty(dim, dtype=np.int32)
    H_goal[:] = UNEXPLORED
    H_start = np.empty(dim, dtype=np.int32)
    H_start[:] = UNEXPLORED
    

    # compute heuristics towards start and goal
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print('THREADS PER BLOCK: ',threadsperblock)
    print('BLOCKS PER GRID: ', blockspergrid)
    print('----- Computing Heuristics -----')
    computeHeuristics[blockspergrid, threadsperblock](grid, start, goal, H_start, H_goal)
    print(blockspergrid)

    # prepare grid blocking guide
    block = np.zeros(dim, dtype=np.int32)
    block = blockshaped(block, TPB, TPB)
    for i in range(block.shape[0]):
        block[i,:] = i
    block = unblockshaped(block, width, height)

    # prepare padded grid, guide, H_goal
    padded_grid = np.zeros((width+2, height+2), dtype=np.int32)
    padded_guide = np.empty((width+2, height+2), dtype=np.int32)
    padded_guide[:] = -1
    padded_block = np.empty((width+2, height+2), dtype=np.int32)
    padded_block[:] = -1
    padded_H_goal = np.empty((width+2, height+2), dtype=np.int32)
    padded_H_goal[:] = UNEXPLORED 
    padGrid[blockspergrid, threadsperblock](grid, padded_grid)
    padGrid[blockspergrid, threadsperblock](guide, padded_guide)
    padGrid[blockspergrid, threadsperblock](block, padded_block)
    padGrid[blockspergrid, threadsperblock](H_goal, padded_H_goal)

    # print(padded_grid)
    # print(padded_guide)
    # print(padded_H_goal)

    # prepare local grids
    grid_blocks = view_as_windows(padded_grid, (TPB+2, TPB+2), step=TPB)
    grid_blocks = grid_blocks.reshape(grid_blocks.shape[0]*grid_blocks.shape[1], grid_blocks.shape[2], grid_blocks.shape[3])

    guide_blocks = view_as_windows(padded_guide, (TPB+2, TPB+2), step=TPB)
    guide_blocks = guide_blocks.reshape(guide_blocks.shape[0]*guide_blocks.shape[1], guide_blocks.shape[2], guide_blocks.shape[3])

    H_goal_blocks = view_as_windows(padded_H_goal, (TPB+2, TPB+2), step=TPB)
    H_goal_blocks = H_goal_blocks.reshape(H_goal_blocks.shape[0]*H_goal_blocks.shape[1], H_goal_blocks.shape[2], H_goal_blocks.shape[3])

    blocks = view_as_windows(padded_block, (TPB+2, TPB+2), step=TPB)
    blocks = blocks.reshape(blocks.shape[0]*blocks.shape[1], blocks.shape[2], blocks.shape[3])
    # print(grid_blocks.shape)
    

    print('Start: ', start)
    print('Goal: ', goal)
    print('Grid')
    print(grid)
    print('Grid Index Guide: ')
    print(guide)
    print('Start H: ')
    print(H_start)
    print('Goal H: ')
    print(H_goal)
    print('Grid Blocking:')
    print(block)
    print('Padded Grid Blocks:')
    print(grid_blocks)
    print('Padded Guide Blocks:')
    print(guide_blocks)
    print('Padded Goal H Blocks:')
    print(H_goal_blocks)
    print('Padded Block Blocks:')
    print(blocks)

    # parents array contains info where tiles came from
    parents = np.empty((width, height, TPB+2, TPB+2), np.int32)
    parents[:] = -1

    # print(parents)

    # Simultaneous local search
    s = timer()
    counter = np.zeros(dim, np.int32)
    # GridDecompSearch[blockspergrid, threadsperblock](grid, start, goal, H_goal, block, parents, grid_blocks, guide_blocks, H_goal_blocks, blocks)
    GridDecompSearch[blockspergrid, threadsperblock](grid, H_goal, block, grid_blocks, start, goal, parents, H_goal_blocks, guide_blocks, blocks, counter)
    print(parents)
    print(counter)
    e = timer()
    print('kernel launch (+ compilation) done in ', e-s, 's')

if __name__ == "__main__":
    main()

    


