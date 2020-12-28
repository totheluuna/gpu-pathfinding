from random import randint, seed
import cv2 as cv
import sys
import os
import math
import numpy as np
import config

seed(config.seed)

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
    return image

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
    width, height = config.dim
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
    path = path.reverse()

def reconstructPathV3(parents, guide, goal_1d_index, path):
    # convert 1D goal index -> 2D goal index
    width, height = parents.shape
    goal_x = int((goal_1d_index-(goal_1d_index%width))/width)
    goal_y = goal_1d_index%width 
    current = (goal_x, goal_y)
    current_1d_index = goal_1d_index

    # print('2D: ', current, '1D: ', current_1d_index)
    # print(guide)
    # print(np.arange(padded_TPB*padded_TPB).reshape(padded_TPB, padded_TPB).astype(np.int32))
    # print(parents)

    ctr = 0
    while current_1d_index != parents[current]:
        # in case of infinite loop
        if ctr > 10:
            print('Timeout!')
            break
        # path.append(current_1d_index)
        parent_1d_index = parents[current]
        current_x = int((parent_1d_index-(parent_1d_index%width))/width)
        current_y = parent_1d_index%width 
        current_1d_index = current_x * width + current_y
        current = (current_x, current_y)
        path.append(guide[current])
        ctr += 1
    # path.append(guide[current])
    path = path.reverse()

def passable(grid, tile):
    x,y = tile
    return grid[x,y] == 1

def drawGrid(grid, start, goal, cost=None, path=None):
    width, height = grid.shape
    print("----- Reconstructing Grid as Text File -----")
    gridStr = ""
    for x in range(width):
        # print(y)
        for y in range(height):
            if passable(grid, (x,y)):
                if (x, y) == start:
                    gridStr += 'S'
                elif (x,y) == goal:
                    gridStr += 'G'
                else:
                    gridStr += '- '
            elif not passable(grid, (x,y)):
                gridStr += '# '
        gridStr += '\n'
    text_file = open("map.txt", "w")
    n = text_file.write(gridStr)
    text_file.close()
