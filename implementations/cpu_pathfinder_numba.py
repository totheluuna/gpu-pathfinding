from random import randint, seed

import cv2 as cv
import sys
import os
import math
import numpy as np

import heapq

from numba import jit, njit, cuda
from numba.typed import List

from timeit import default_timer as timer

seed(1)
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
@njit
def passable(grid, tile):
    x,y = tile
    return grid[tile] == 1
@njit
def inBounds(grid, tile):
    (x, y) = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
@njit
def getNeighbors(grid, tile):
    (x, y) = tile
    results = []
    possibleNeighbors = [(x+1,y), (x,y-1), (x-1,y), (x,y+1)]
    for tile in possibleNeighbors:
        if inBounds(grid, tile):
            if passable(grid, tile):
                results.append(tile)
    if (x + y)%2 == 0: results.reverse()
    return results
@njit
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)

@njit
def search(grid, start, goal, parentHash, FValue):
    width, height = grid.shape

    temp_data = (-1, (-1, -1)) 

    openList = []
    openList.append(temp_data)
    openListEntryFinder = {temp_data[1] : temp_data}
    openList.remove(temp_data)
    openListEntryFinder.pop(temp_data[1])

    closedList = []
    closedList.append(temp_data)
    closedListEntryFinder = {temp_data[1] : temp_data}
    closedList.remove(temp_data)
    closedListEntryFinder.pop(temp_data[1])
    
    GValue = np.zeros((width, height), dtype=np.int64)
    HValue = np.zeros((width, height), dtype=np.int64)
    parentHash[:] = np.array([-1,-1])
    
    addToPQ(openList, openListEntryFinder, start, np.int64(0))

    startX, startY = start
    GValue[startX, startY] = 0
    HValue[startX, startY] = heuristic(start, goal)
    FValue[startX, startY] = GValue[startX, startY] + HValue[startX, startY]

    while not len(openList) == 0:
        current = popFromPQ(openList, openListEntryFinder)
        # print(grid[current])
        currentX, currentY = current
        if current == goal:
            # print('Found goal %s' %(str(current)))
            break
        for next in getNeighbors(grid, current):
            # print(next)
            nextX, nextY = next
            newG = GValue[currentX, currentY] + 1 # constant 1 since grid
            if next in openListEntryFinder:
                if newG < GValue[nextX, nextY]:
                    removeFromPQ(openList, openListEntryFinder, next)
            if next in closedListEntryFinder:
                if newG < GValue[nextX, nextY]:
                    removeFromPQ(closedList, closedListEntryFinder, next)
            if (next not in openListEntryFinder) and (next not in closedListEntryFinder):
                parentHash[nextX, nextY] = np.array([currentX, currentY])
                GValue[nextX, nextY] = newG
                HValue[nextX, nextY] = heuristic(next, goal)
                FValue[nextX, nextY] = GValue[nextX, nextY] + HValue[nextX, nextY]
                addToPQ(openList, openListEntryFinder, next, FValue[nextX, nextY])
        addToPQ(closedList, closedListEntryFinder, current, FValue[currentX, currentY])

# functions for priority queue
@njit
def addToPQ(elements, entryFinder, item, priority):
    if item in entryFinder:
        removeFromPQ(elements, entryFinder, item)
    entry = (priority, item)
    entryFinder[item] = entry
    heapq.heappush(elements, entry)
@njit
def removeFromPQ(elements, entryFinder, item):
    entry = entryFinder.pop(item)
    elements.remove(entry)
    heapq.heapify(elements)
@njit
def popFromPQ(elements, entryFinder):
    priority, item = heapq.heappop(elements)
    return item

@cuda.jit
def GPUSampleKernel(grid, start, goal, hArray):
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid
    print(bpg)
    if x < grid.shape[0] and y < grid.shape[1]:
        goal_x, goal_y = goal
        if grid[x, y] != 0:
            hArray[x, y] = heuristic(start, goal)


def main():
    # create grid from image dataset
    scale_factor = 4 # scales to a power of 2
    dim = (int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor)))
    grid = np.zeros(dim, dtype=np.int32)
    createGridFromDatasetImage('dataset/da2-png', grid, dim)
    print(grid)
    # generate random start and goal
    start = [-1, -1]
    goal = [-1, -1]
    randomStartGoal(grid, start, goal)
    start = tuple(start)
    goal = tuple(goal)
    
    # search for path
    width, height = grid.shape
    parents = np.empty((width, height, 2), dtype=np.int64)
    cost = np.zeros((width, height), dtype=np.int64)
    s = timer()
    search(grid, start, goal, parents, cost)
    # print(parents)
    # print(cost)
    # reconstruct path
    path = []
    reconstructPathV2(parents, tuple(start), tuple(goal), path)
    e = timer()
    print('Before compilation: ', e-s)
    s = timer()
    search(grid, start, goal, parents, cost)
    # print(parents)
    # print(cost)
    # reconstruct path
    path = []
    reconstructPathV2(parents, tuple(start), tuple(goal), path)
    e = timer()
    print('After compilation: ', e-s)
    print(path)

    # GPU Pathfinder
    hArray = np.zeros(grid.shape, dtype=np.int64)
    TPB = 16
    path = []
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(grid.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(grid.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    GPUSampleKernel[blockspergrid, threadsperblock](grid, start, goal, hArray)
    print(hArray)


if __name__ == "__main__":
    main()

    


