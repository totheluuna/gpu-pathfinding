from random import randint, seed
from models import grid, weighted_grid

import cv2 as cv
import sys

seed(69)

# Creates a grid object with random walls
# using x and y as dimension parameters
# Returns a grid object
def createGrid(x, y, algorithm):
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
    for y in range(grid.height):
        for x in range(grid.width):
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
def reconstructPath(cameFrom, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = cameFrom[current]
    path.append(start)
    path.reverse()
    return path

# TO DO: IMPLEMENT FUNCTION TO CONVERT IMAGE TO GRID OBJECT
def imageToGrid(image):
    print("----- Converting Image to Grid Object -----")
    # load image
    img = cv.imread(cv.samples.findFile(image))
    if img is None:
        sys.exit("Could not read the image.")

    # resize image
    print('Original Dimensions : ',img.shape)
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA) 
    print('Resized Dimensions : ',resized.shape) 

    # convert to grayscale
    grayImg = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    cv.imwrite("gray_image.png", grayImg)

    # calculate histogram
    # hist = cv.calcHist([grayImg],[0],None,[256],[0,256])
    # for i in range(len(hist)):
    #     if hist[i][0] > 0:
    #         print(i)
    
    # apply thresholding to easily separate walkable and unwalkable areas
    ret, thresh1 = cv.threshold(grayImg,75,229,cv.THRESH_BINARY)
    print('Threshold Dimensions : ', thresh1.shape)

    # add unwalkable areas to walls list
    walls = []
    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            # if the current value = 0 (meaning black) append to list of walls
            if thresh1[i][j] == 0:
                walls.append((i,j))
    # print(walls)

    # check if walls < number of pixels in image
    def checkWalls(walls):
        return len(walls) > thresh1.shape[0]*thresh1.shape[1]

    print("All walls? ", checkWalls(walls))
    cv.imwrite("threshold.png", thresh1)

    # initialize grid
    x = thresh1.shape[0]
    y = thresh1.shape[1]
    outputGrid = weighted_grid.WeightedGrid(x,y)
    outputGrid.walls = walls

    return outputGrid

     