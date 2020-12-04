from random import randint, seed

import cv2 as cv
import sys
import os
import math
import numpy as np

import heapq
from timeit import default_timer as timer

from numba import cuda

OPEN = 1
CLOSED = 0
UNEXPLORED = 999999

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
@cuda.jit(device=True)
def passable(grid, tile):
    x,y = tile
    return grid[x,y] == 1
@cuda.jit(device=True)
def inBounds(grid, tile):
    x, y = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
@cuda.jit(device=True)
def getNeighbors(grid, tile, neighbors):
    # TO DO: modify to use numpy array
    (x, y) = tile
    for i in range(neighbors.size):
        if (x+y)%2 == 0:
            if i == 0:
                neighbors[i,0] = x
                neighbors[i,1] = y+1
            elif i == 1:
                neighbors[i,0] = x-1
                neighbors[i,1] = y
            elif i == 2:
                neighbors[i,0] = x
                neighbors[i,1] = y-1
            elif i == 3:
                neighbors[i,0] = x+1
                neighbors[i,1] = y
        else:
            if i == 0:
                neighbors[i,0] = x+1
                neighbors[i,1] = y
            elif i == 1:
                neighbors[i,0] = x
                neighbors[i,1] = y-1
            elif i == 2:
                neighbors[i,0] = x-1
                neighbors[i,1] = y
            elif i == 3:
                neighbors[i,0] = x
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
def search(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors):
    width, height = grid.shape
    start_x, start_y = start
    goal_x, goal_y = goal

    open[start_x, start_y] = 0
    g[start_x, start_y] = 0
    h[start_x, start_y] = heuristic(start, goal)
    cost[start_x, start_y] = g[start_x, start_y] + h[start_x, start_y]

    counter = 0
    # while np.amin(open) < UNEXPLORED:
    while getMin(open) < UNEXPLORED:
        current_x, current_y = getMinIndex(open)
        current = (current_x, current_y)
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
                    parents[next_x, next_y, 0] = current_x
                    parents[next_x, next_y, 1] = current_y
                    g[next_x, next_y] = new_g
                    h[next_x, next_y] = heuristic(next, goal)
                    cost[next_x, next_y] = g[next_x, next_y] + h[next_x, next_y]
                    open[next_x, next_y] = cost[next_x, next_y]
                closed[current_x, current_y] = cost[current_x, current_y]
                open[current_x, current_y] = UNEXPLORED
        counter += 1

@cuda.jit
def GPUPathfinder(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors, counter):
    x, y = cuda.grid(2)
    width, height = cost.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid
    # print(bpg)
    if x < grid.shape[0] and y < grid.shape[1]:
        # do the search for as many times as number of tiles in the grid
        search(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors)
        # counter += 1
        # cuda.syncthreads() 

def main():
    # create grid from image dataset
    scale_factor = 4 # scales to a power of 2
    dim = (int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor)))
    UNEXPLORED = int(math.pow(2, (scale_factor*2)))
    grid = np.zeros(dim, dtype=np.int32)
    createGridFromDatasetImage('dataset/da2-png', grid, dim)
    print(grid)
    
    # generate random start and goal
    start = [-1, -1]
    goal = [-1, -1]
    neighbors = np.empty((4,2), dtype=np.int32)
    neighbors[:] = np.array([0,0])
    print(neighbors)
    randomStartGoal(grid, start, goal)
    start = np.array(start)
    goal = np.array(goal)
    print(start)
    print(goal)

    # search for path
    width, height = grid.shape
    open = np.empty((width, height), dtype=np.int32) # open or closed
    open[:] = UNEXPLORED
    closed = np.empty((width, height), dtype=np.int32) # open or closed
    closed[:] = UNEXPLORED
    parents = np.empty((width, height, 2), dtype=np.int32)
    parents[:] = np.array([-1,-1])
    cost = np.zeros((width, height), dtype=np.int32)
    g = np.zeros((width, height), dtype=np.int32)
    h = np.zeros((width, height), dtype=np.int32)
    x,y = start
    # print(parents)
    # print("----- Searching for Path -----")
    # s = timer()
    # search(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors)
    # x,y = start
    # path = []
    # reconstructPathV2(parents, tuple(start), tuple(goal), path)
    # e = timer()
    # print('(Search + compilation) Path found in ', e-s, 's')
    # s = timer()
    # search(grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors)
    # x,y = start
    # path = []
    # reconstructPathV2(parents, tuple(start), tuple(goal), path)
    # e = timer()
    # print('(Post-compilation) Path found in ', e-s, 's')
    # print(path)
    TPB = 16
    path = []
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    counter = 0
    GPUPathfinder[blockspergrid, threadsperblock](grid, start, goal, open, closed, parents, cost, g, h, UNEXPLORED, neighbors, counter)
    x,y = start
    path = []
    reconstructPathV2(parents, tuple(start), tuple(goal), path)
    # e = timer()
    # print('(Post-compilation) Path found in ', e-s, 's')
    print(path)


if __name__ == "__main__":
    main()

    


