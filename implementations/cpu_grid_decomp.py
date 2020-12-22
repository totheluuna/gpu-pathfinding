from random import randint, seed

import cv2 as cv
import sys
import os
import math
import numpy as np

import heapq
from timeit import default_timer as timer
from skimage.util.shape import view_as_windows

OPEN = 1
CLOSED = 0
UNEXPLORED = 999999

scale_factor = 3 # scales to a power of 2
dim = (int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor)))
UNEXPLORED = int(math.pow(2, (scale_factor*2)))
TPB = 4

# from numba import jit
# from numba.typed import List

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
# @jit
def passable(grid, tile):
    x,y = tile
    # return grid[x,y] == 1
    try:
        return grid[x,y] == 1
    except :
        return 0
# @jit
def inBounds(grid, tile):
    (x, y) = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
# @jit
def getNeighbors(grid, tile, neighbors):
    # TO DO: modify to use numpy array
    (x, y) = tile
    for i in range(neighbors.size):
    #     if (x+y)%2 == 0:
    #         if i == 0:
    #             neighbors[i,0] = x
    #             neighbors[i,1] = y+1
    #         elif i == 1:
    #             neighbors[i,0] = x-1
    #             neighbors[i,1] = y
    #         elif i == 2:
    #             neighbors[i,0] = x
    #             neighbors[i,1] = y-1
    #         elif i == 3:
    #             neighbors[i,0] = x+1
    #             neighbors[i,1] = y
    #         
    #     else:
    #         if i == 0:
    #             neighbors[i,0] = x+1
    #             neighbors[i,1] = y
    #         elif i == 1:
    #             neighbors[i,0] = x
    #             neighbors[i,1] = y-1
    #         elif i == 2:
    #             neighbors[i,0] = x-1
    #             neighbors[i,1] = y
    #         elif i == 3:
    #             neighbors[i,0] = x
    #             neighbors[i,1] = y+1
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

    # results = []
    # possibleNeighbors = [(x+1,y), (x+1,y-1), (x,y-1), (x-1,y-1), (x-1,y), (x-1,y+1), (x,y+1), (x+1,y+1)]
    # for tile in possibleNeighbors:
    #     if inBounds(grid, tile):
    #         if passable(grid, tile):
    #             results.append(tile)
    #     # if inBounds(grid, tile) and passable(grid, tile):
    #     #     results.append(tile)
    # # if (x + y)%2 == 0: results.reverse()
    # return results
# @jit
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)
    return int(math.pow((x1-x2),2) + math.pow((y1-y2),2))

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



# @jit(nopython=True)
def search(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors):
    width, height = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal

    open[start_x, start_y] = 0
    g[start_x, start_y] = 0
    h[start_x, start_y] = heuristic(start, goal)
    cost[start_x, start_y] = g[start_x, start_y] + h[start_x, start_y]

    counter = 0
    while np.amin(open) < UNEXPLORED:
        # print("\riterations: {}".format(counter), end='')
        # current = np.unravel_index(np.argmin(open, axis=None), open.shape)
        # current_x, current_y = current
        current_x, current_y = getMinIndex(open)
        current = (current_x, current_y)
        if current_x == goal_x and current_y == goal_y:
            print("\riterations: {}".format(counter), end='')
            break
        getNeighbors(grid, current, neighbors)
        # for next in getNeighbors(grid, current, neighbors):
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
                    # parents[next_x, next_y] = np.array([current_x, current_y])
                    parents[next_x, next_y] = current_x * width + current_y
                    g[next_x, next_y] = new_g
                    h[next_x, next_y] = heuristic(next, goal)
                    cost[next_x, next_y] = g[next_x, next_y] + h[next_x, next_y]
                    open[next_x, next_y] = cost[next_x, next_y]
                closed[current_x, current_y] = cost[current_x, current_y]
                open[current_x, current_y] = UNEXPLORED
        counter += 1
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

def padGrid(grid, padded_grid):
    width, height = dim

    for i in range(width):
        for j in range(height):
            padded_grid[i+1, j+1] = grid[i,j]

def computeHeuristics(grid, tile, h):
    width, height = dim

    for i in range(width):
        for j in range(height):
            h[i,j] = heuristic((i,j), tile)

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

    # blocking index guide
    block = np.zeros(dim, dtype=np.int32)
    block = blockshaped(block, TPB, TPB)
    for i in range(block.shape[0]):
        block[i,:] = i
    block = unblockshaped(block, width, height)

    h = np.empty(dim, dtype=np.int32)
    h[:] = UNEXPLORED
    computeHeuristics(grid, goal, h)

    
    padded_grid = np.zeros((width+2, height+2), dtype=np.int32)
    padded_guide = np.empty((width+2, height+2), dtype=np.int32)
    padded_guide[:] = -1
    padded_block = np.empty((width+2, height+2), dtype=np.int32)
    padded_block[:] = -1
    padded_h = np.empty((width+2, height+2), dtype=np.int32)
    padded_h[:] = UNEXPLORED 
    padGrid(grid, padded_grid)
    padGrid(guide, padded_guide)
    padGrid(block, padded_block)
    padGrid(h, padded_h)

    # print(padded_grid)
    # print(padded_guide)
    # print(padded_h)

    # prepare local grids
    grid_blocks = view_as_windows(padded_grid, (TPB+2, TPB+2), step=TPB)
    grid_blocks = grid_blocks.reshape(grid_blocks.shape[0]*grid_blocks.shape[1], grid_blocks.shape[2], grid_blocks.shape[3])

    guide_blocks = view_as_windows(padded_guide, (TPB+2, TPB+2), step=TPB)
    guide_blocks = guide_blocks.reshape(guide_blocks.shape[0]*guide_blocks.shape[1], guide_blocks.shape[2], guide_blocks.shape[3])

    h_blocks = view_as_windows(padded_h, (TPB+2, TPB+2), step=TPB)
    h_blocks = h_blocks.reshape(h_blocks.shape[0]*h_blocks.shape[1], h_blocks.shape[2], h_blocks.shape[3])

    blocks = view_as_windows(padded_block, (TPB+2, TPB+2), step=TPB)
    blocks = blocks.reshape(blocks.shape[0]*blocks.shape[1], blocks.shape[2], blocks.shape[3])
    print(grid_blocks.shape)

    print('Start: ', start)
    print('Goal: ', goal)
    print('Grid')
    print(grid)
    print('Grid Index Guide: ')
    print(guide)
    print('H (from goal): ')
    print(h)
    print('Grid Blocking: ')
    print(block)
    print('Padded Grid Blocks: ', grid_blocks.shape)
    print(grid_blocks[0])
    print('Padded Guide Blocks: ', guide_blocks.shape)
    print(guide_blocks[0])
    print('Padded Goal H Blocks:', h_blocks.shape)
    print(h_blocks[0])
    print('Padded Block Guide Blocks: ', blocks.shape)
    print(blocks[0])

    # # initialize essential arrays for search algorithm
    # print('----- Initializing Variables -----')


    neighbors = np.empty((8,2), dtype=np.int32)
    neighbors[:] = np.array([0,0])
    print(neighbors)
    
    open = np.empty((TPB+2, TPB+2), dtype=np.int32) # open or closed
    open[:] = UNEXPLORED
    closed = np.empty((TPB+2, TPB+2), dtype=np.int32) # open or closed
    closed[:] = UNEXPLORED
    parents = np.empty((TPB+2, TPB+2), dtype=np.int32)
    # parents[:] = np.array([-1,-1])
    parents[:] = -1
    cost = np.zeros((TPB+2, TPB+2), dtype=np.int32)
    g = np.zeros((TPB+2, TPB+2), dtype=np.int32)
    x,y = start
    # print(parents)

    print("----- Searching for Path -----")
    s = timer()
    search(grid_blocks[0], start, goal, open, closed, parents, cost, g, h_blocks[0], UNEXPLORED, neighbors)
    x,y = start
    print(parents)
    # path = []
    # reconstructPathV2(parents, tuple(start), tuple(goal), path)
    e = timer()
    print('\nPath found in ', e-s, 's')
    # print(path)

if __name__ == "__main__":
    main()

    


