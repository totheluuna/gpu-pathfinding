from random import randint, seed
import argparse
import cv2 as cv
import sys
import os
import math
import helper
import config
import numpy as np

import cpu_search as cpu
import gpu_search as gpu

def test_func():
    print('scale factor: ', config.scale_factor)
    print('TPB: ', config.TPB)
    print('max value: ', config.UNEXPLORED)

def main():

    parser = argparse.ArgumentParser(description='GPU Pathfinding')
    parser.add_argument('scale_factor', type=int, help='Scale factor (power of 2)')
    parser.add_argument('TPB', type=int, help='Block width')
    args = parser.parse_args()
    config.scale_factor = args.scale_factor
    config.TPB = args.TPB
    config.dim = int(math.pow(2, config.scale_factor)), int(math.pow(2, config.scale_factor))
    config.UNEXPLORED = int(math.pow(2, (config.scale_factor*2)))

    width, height = config.dim
    test_func()

    print('----- Preparing Grid -----')
    # create grid from image dataset
    # grid = np.zeros(config.dim, dtype=np.int32)
    # createGridFromDatasetImage('dataset/da2-png', grid, config.dim)
    grid = np.ones(config.dim, dtype=np.int32)

    # generate random start and goal
    # start = [-1, -1]
    # goal = [-1, -1]
    # randomStartGoal(grid, start, goal)
    start = [0, 0]
    goal = [grid.shape[0]-1, grid.shape[1]-1]
    start = np.array(start)
    goal = np.array(goal)
    
    print(grid)
    print(start)
    print(goal)

    # cpu implementation
    # cpu.test(grid, start, goal)

    # gpu implementation
    gpu.test(grid, start, goal)


    

if __name__ == "__main__":
    main()