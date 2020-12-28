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
    parser = argparse.ArgumentParser(description='CPU vs GPU Pathfinding')
    parser.add_argument('scale_factor', type=int, help='Scale factor (power of 2)')
    parser.add_argument('TPB', type=int, help='Block width')
    parser.add_argument('complexity', type=str, help='Map Complexity')
    args = parser.parse_args()
    config.scale_factor = args.scale_factor
    config.TPB = args.TPB
    config.padded_TPB = config.TPB + 2
    config.dim = int(math.pow(2, config.scale_factor)), int(math.pow(2, config.scale_factor))
    config.UNEXPLORED = int(math.pow(2, (config.scale_factor*2)))
    complexity = args.complexity

    width, height = config.dim
    test_func()

    print('----- Preparing Grid -----')
    # create grid from image dataset
    grid = np.zeros(config.dim, dtype=np.int32)
    # helper.createGridFromDatasetImage('dataset/select-maps/simplest', grid, config.dim)
    image = helper.createGridFromDatasetImage('dataset/select-maps/%s'%(complexity), grid, config.dim)
    # grid = np.ones(config.dim, dtype=np.int32)

    # generate random start and goal
    start = [-1, -1]
    goal = [-1, -1]
    helper.randomStartGoal(grid, start, goal)
    # start = [0, 0]
    # goal = [grid.shape[0]-1, grid.shape[1]-1]
    start = np.array(start)
    goal = np.array(goal)
    
    print(grid)
    print(start)
    print(goal)

    helper.drawGrid(grid, tuple(start), tuple(goal))

    # cpu implementation
    runs_cpu, time_ave_cpu, path_cpu = cpu.test(grid, start, goal)
    # gpu implementation
    runs_gpu, time_ave_gpu, path_gpu = gpu.test(grid, start, goal)

    print('----- Summary -----')
    print('Image used:', image)
    print('Start:', start)
    print('Goal:', goal)
    print()
    print('Average runtime in', runs_cpu, 'runs (CPU):', time_ave_cpu)
    print('path length (CPU):', len(path_cpu))
    print()
    print('Average runtime in', runs_gpu, 'runs (GPU):', time_ave_gpu)
    print('path length (GPU):', len(path_gpu))
    print()
    print('full path (CPU): ', path_cpu)
    print()
    print('full path (GPU): ', path_gpu)


if __name__ == "__main__":
    main()