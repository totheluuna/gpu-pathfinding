from random import randint, seed

import cv2 as cv
import sys
import os
import math
import numpy as np

import heapq

from numba import jit

seed(1)
# functions for converting images to grids
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
def imageToGrid(image):
    print("----- Converting Image to Grid Object -----")
    # load image
    img = cv.imread(cv.samples.findFile(image))
    if img is None:
        sys.exit("Could not read the image.")
    cv.imwrite("output/original.png", img)

    # resize image
    print('Original Dimensions : ',img.shape)
    # scale_percent = 10 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    scale_power = 7 # resize to a power of 2
    width = int(math.pow(2, scale_power))
    height = int(math.pow(2, scale_power))
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    cv.imwrite("output/resized.png", resized) 
    print('Resized Dimensions : ',resized.shape) 

    # convert to grayscale
    grayImg = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    cv.imwrite("output/grayscale.png", grayImg)

    # calculate histogram
    # hist = cv.calcHist([grayImg],[0],None,[256],[0,256])
    # for i in range(len(hist)):
    #     if hist[i][0] > 0:
    #         print(i)
    
    # apply thresholding to easily separate walkable and unwalkable areas
    ret, thresh1 = cv.threshold(grayImg,75,229,cv.THRESH_BINARY)
    cv.imwrite("output/threshold.png", thresh1)
    print('Threshold Dimensions : ', thresh1.shape)

    gridArray = np.ones((thresh1.shape), dtype=np.int32)
    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            # if the current value = 0 (meaning black) append to list of walls
            if thresh1[i][j] == 0:
                gridArray[i,j] = 0
            else:
                gridArray[i,j] = 1

    return gridArray
def randomStartGoal(grid):
    print('----- Generating Random Start and Goal -----')
    dist = 0
    width, height = grid.shape
    hypotenuse = math.sqrt(math.pow(width, 2) + math.pow(height,2))
    while dist <= 0.50*hypotenuse:
        random_x = randint(0, width-1)
        random_y = randint(0, height-1)
        start = (random_x, random_y)
        while grid[random_x, random_y] == 0:
            random_x = randint(0, width-1)
            random_y = randint(0, height-1)
            start = (random_x, random_y)

        random_x = randint(0, width-1)
        random_y = randint(0, height-1)
        goal = (random_x, random_y)
        while grid[random_x, random_y] == 0:
            random_x = randint(0, width-1)
            random_y = randint(0, height-1)
            goal = (random_x, random_y)
        a = np.array(start)
        b = np.array(goal)
        dist = np.linalg.norm(a-b)

    return start, goal
def createGridFromDatasetImage(dataset):
    print('----- Creating Grid Object from Dataset Image-----')
    listOfImages = getListOfFiles(dataset)
    image = listOfImages[randint(0, len(listOfImages)-1)]
    print('Random Image: ', image)
    grid = imageToGrid(image)

    return grid

# function for reconstructing found path
def reconstructPathV2(cameFrom, start, goal):
    currentX, currentY = goal
    path = []
    while (currentX, currentY) != start:
        path.append((currentX, currentY))
        currentX, currentY = cameFrom[currentX, currentY]
    path.append(start)
    path.reverse
    return path

# functions for pathfinding
def passable(grid, tile):
    x,y = tile
    return grid[tile] == 1
def inBounds(grid, tile):
    (x, y) = tile
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
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
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)
def search(grid, start, goal):
    width, height = grid.shape

    openList = []
    openListEntryFinder = {}

    closedList = []
    closedListEntryFinder = {}
    
    parentHash = np.empty((width, height, 2), dtype=np.int32)
    GValue = np.zeros((width, height), dtype=np.int32)
    HValue = np.zeros((width, height), dtype=np.int32)
    FValue = np.zeros((width, height), dtype=np.int32)
    parentHash[:] = np.array([-1,-1])
    
    addToPQ(openList, openListEntryFinder, start, 0)
    startX, startY = start
    GValue[startX, startY] = 0
    HValue[startX, startY] = heuristic(start, goal)
    FValue[startX, startY] = GValue[startX, startY] + HValue[startX, startY]

    while not len(openList) == 0:
        current = popFromPQ(openList, openListEntryFinder)
        print(grid[current])
        currentX, currentY = current
        if current == goal:
            print('Found goal %s' %(str(current)))
            break
        for next in getNeighbors(grid, current):
            # print(next)
            nextX, nextY = next
            newG = GValue[currentX, currentY] + 1 # constant 1 since grid
            if next in openListEntryFinder:
                if newG < GValue[nextX, nextY]:
                    removeFromPQ(openListEntryFinder, next)
            if next in closedListEntryFinder:
                if newG < GValue[nextX, nextY]:
                    removeFromPQ(closedListEntryFinder, next)
            if (next not in openListEntryFinder) and (next not in closedListEntryFinder):
                parentHash[nextX, nextY] = np.array([currentX, currentY])
                GValue[nextX, nextY] = newG
                HValue[nextX, nextY] = heuristic(next, goal)
                FValue[nextX, nextY] = GValue[nextX, nextY] + HValue[nextX, nextY]
                addToPQ(openList, openListEntryFinder, next, FValue[nextX, nextY])
        addToPQ(closedList, closedListEntryFinder, current, FValue[currentX, currentY])

    return parentHash, FValue

# functions for priority queue
def addToPQ(elements, entryFinder, item, priority=0):
    if item in entryFinder:
        removeFromPQ(item)
    entry = [priority, item]
    entryFinder[item] = entry
    heapq.heappush(elements, entry)
def removeFromPQ(entryFinder, item):
    REMOVED = (9999, 9999)
    entry = entryFinder.pop(item)
    entry[-1] = REMOVED
def popFromPQ(elements, entryFinder):
    REMOVED = (9999, 9999)
    priority, item = heapq.heappop(elements)
    if item is not REMOVED:
        del entryFinder[item]
        return item
    raise KeyError('pop from an empty priority queue')

def main():
    # create grid from image dataset
    grid = createGridFromDatasetImage('dataset/da2-png')
    print(grid)
    # generate random start and goal
    start, goal = randomStartGoal(grid)
    print(start, goal)
    # search for path
    parents, cost = search(grid, start, goal)
    # reconstruct path
    path = reconstructPathV2(parents, start, goal)
    print(path)


if __name__ == "__main__":
    main()

    


