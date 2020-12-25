from random import randint, seed
import argparse
import cv2 as cv
import sys
import os
import math

scale_factor = 4 # scales to a power of 2
dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
TPB = 4
padded_TPB = TPB + 2

UNEXPLORED = int(math.pow(2, (scale_factor*2)))
# UNEXPLORED = 9999999
OPEN = 1
CLOSED = 0

def test_func():
    print('scale factor: ', scale_factor)
    print('TPB: ', TPB)
    print('max value: ', UNEXPLORED)

def main():
    global scale_factor
    global TPB
    global dim
    global UNEXPLORED

    parser = argparse.ArgumentParser(description='GPU Pathfinding')
    parser.add_argument('scale_factor', type=int, help='Scale factor (power of 2)')
    parser.add_argument('TPB', type=int, help='Block width')
    args = parser.parse_args()
    scale_factor = args.scale_factor
    TPB = args.TPB
    dim = int(math.pow(2, scale_factor)), int(math.pow(2, scale_factor))
    UNEXPLORED = int(math.pow(2, (scale_factor*2)))
    test_func()

    

if __name__ == "__main__":
    main()