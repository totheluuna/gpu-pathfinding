from random import randint, seed
from models import grid, weighted_grid

import cv2 as cv
import sys
import os
import math
import numpy as np

import config

# seed(420696969)
seed(1)

def createGrid(x, y, algorithm):
    # Creates a grid object with random walls
    # using x and y as dimension parameters
    # Returns a grid object

    # initialize grid
    if algorithm == 'breadth_first':
        outputGrid = grid.Grid(x, y)
    else:
        outputGrid = weighted_grid.WeightedGrid(x,y) 

    # initialize a random list of walls
    # so as to not fill the whole grid with walls
    # walls list must have length = (xy)/2
    N = int((x*y)/8) 
    randomWalls = [0]*N
    # print(len(randomWalls))

    for i in range(N):
        # change the initial value to a tuple with random (x,y)
        # such that it is within the dimensions of the grid
        random_x = randint(0, x-1)
        random_y = randint(0, y-1)
        while((random_x, random_y) in randomWalls):
            random_x = randint(0, x-1)
            random_y = randint(0, y-1)
        randomWalls[i] = (random_x, random_y)
    # print(randomWalls)
    outputGrid.walls = randomWalls

    return outputGrid

def drawGrid(grid, start, goal, cost=None, path=None):
    gridStr = ""
    for x in range(grid.height):
        for y in range(grid.width):
            if grid.passable((x,y)):
                if cost is not None:
                    print('%s ' % str(cost[(x,y)]),end='')
                if path and (x,y) in path:
                    if (x,y) == start:
                        print('S ',end='')
                    elif (x,y) == goal:
                        print('G ',end='')
                    else:
                        print('@ ',end='')
                else:
                    print('- ',end='')
            elif not grid.passable((x,y)):
                print('# ',end='')
        print('')

def drawGrid2(grid, start, goal, cost=None, path=None):
    print("----- Reconstructing Grid as Text File -----")
    gridStr = ""
    for y in range(grid.height):
        # print(y)
        for x in range(grid.width):
            if grid.passable((x,y)):
                gridStr += '- '
            elif not grid.passable((x,y)):
                gridStr += '# '
        gridStr += '\n'
    text_file = open("sample.txt", "w")
    n = text_file.write(gridStr)
    text_file.close()

def drawGrid3(grid, start, goal, cost=None, path=None):
    print("----- Reconstructing Grid as Text File -----")
    gridStr = ""
    for x in range(grid.height):
        for y in range(grid.width):
            if grid.passable((x,y)):
                if cost is not None:
                    print('%s ' % str(cost[(x,y)]),end='')
                if path and (x,y) in path:
                    if (x,y) == start:
                        # print('S ',end='')
                        gridStr += 'S '
                    elif (x,y) == goal:
                        # print('G ',end='')
                        gridStr += 'G '
                    else:
                        # print('@ ',end='')
                        gridStr +='@ '
                else:
                    # print('- ',end='')
                    gridStr += '- '
            elif not grid.passable((x,y)):
                # print('# ',end='')
                gridStr += '# '
        # print('')
        gridStr += '\n'
    print('See output/reconstructed_path.txt')
    text_file = open("output/reconstructed_path.txt", "w")
    n = text_file.write(gridStr)
    text_file.close()

def reconstructPath(cameFrom, start, goal):
    current = goal
    path = []
    while current != start:
        # print(current)
        path.append(current)
        current = cameFrom[current]
    path.append(start)
    path.reverse()
    return path

def reconstructPathV2(cameFrom, start, goal):
    currentX, currentY = goal
    path = []
    while (currentX, currentY) != start:
        path.append((currentX, currentY))
        currentX, currentY = cameFrom[currentX, currentY]
    path.append(start)
    path.reverse
    return path

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
    scale_power = 4 # resize to a power of 2
    width = int(math.pow(2, scale_power))
    height = int(math.pow(2, scale_power))
    config.dim = (width, height)
    resized = cv.resize(img, config.dim, interpolation = cv.INTER_AREA)
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

    # # calculate histogram
    # hist = cv.calcHist([thresh1],[0],None,[256],[0,256])
    # for i in range(len(hist)):
    #     if hist[i][0] > 0:
    #         print(i)
    # add unwalkable areas to walls list
    walls = []
    gridArray = np.ones((thresh1.shape), dtype=np.int32)
    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            # if the current value = 0 (meaning black) append to list of walls
            if thresh1[i][j] == 0:
                gridArray[i,j] = 0
                walls.append((i,j))
            else:
                gridArray[i,j] = 1
    # print(walls)

    # check if walls < number of pixels in image
    print("All walls? ", len(walls) > thresh1.shape[0]*thresh1.shape[1])

    # initialize grid
    x = thresh1.shape[0]
    y = thresh1.shape[1]
    outputGrid = weighted_grid.WeightedGrid(x,y)
    outputGrid.walls = walls

    return outputGrid, gridArray
        
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

def randomStartGoal(grid):
    print('----- Generating Random Start and Goal -----')
    dist = 0
    hypotenuse = math.sqrt(math.pow(grid.width, 2) + math.pow(grid.height,2))
    while dist <= 0.50*hypotenuse:
        random_x = randint(0, grid.width-1)
        random_y = randint(0, grid.height-1)
        start = (random_x, random_y)
        while start in grid.walls:
            random_x = randint(0, grid.width-1)
            random_y = randint(0, grid.height-1)
            start = (random_x, random_y)

        random_x = randint(0, grid.width-1)
        random_y = randint(0, grid.height-1)
        goal = (random_x, random_y)
        while (goal in grid.walls):
            random_x = randint(0, grid.width-1)
            random_y = randint(0, grid.height-1)
            goal = (random_x, random_y)
        a = np.array(start)
        b = np.array(goal)
        dist = np.linalg.norm(a-b)
    print('start: ', start, ' goal: ', goal)

    return start, goal

def createGridFromDatasetImage(dataset):
    print('----- Creating Grid Object from Dataset Image-----')
    listOfImages = getListOfFiles(dataset)
    image = listOfImages[randint(0, len(listOfImages)-1)]
    print('Random Image: ', image)
    grid, gridArray = imageToGrid(image)

    return grid, gridArray
     