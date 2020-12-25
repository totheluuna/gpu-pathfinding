from random import randint, seed
import argparse
import cv2 as cv
import sys
import os
import math
import helper
import globals
import configobj
import numpy as np

scale_factor = 4 # scales to a power of 2
dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
TPB = 4
padded_TPB = TPB + 2

UNEXPLORED = int(math.pow(2, (scale_factor*2)))
# UNEXPLORED = 9999999
OPEN = 1
CLOSED = 0

def test_func():
    print('scale factor: ', configobj.scale_factor)
    print('TPB: ', configobj.TPB)
    print('max value: ', configobj.UNEXPLORED)

def main():
    # global scale_factor
    # global TPB
    # global dim
    # global UNEXPLORED

    parser = argparse.ArgumentParser(description='GPU Pathfinding')
    parser.add_argument('scale_factor', type=int, help='Scale factor (power of 2)')
    parser.add_argument('TPB', type=int, help='Block width')
    args = parser.parse_args()
    configobj.scale_factor = args.scale_factor
    configobj.TPB = args.TPB
    configobj.dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
    configobj.UNEXPLORED = int(math.pow(2, (scale_factor*2)))

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

    print(grid)
    print(start)
    print(goal)
    print(guide)


    

if __name__ == "__main__":
    main()