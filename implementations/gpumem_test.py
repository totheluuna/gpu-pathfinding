TPB = 4

from numba import cuda, int32
import numpy as np
import cupy as cp
import math

@cuda.jit
def gpu_memory_test(arr):
    x, y = cuda.grid(2)
    width, height = arr.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    shared_arr = cuda.shared.array(shape=(TPB, TPB), dtype=int32)

    for i in range(bpg):
        arr[tx + i * TPB, ty + i * TPB] = i
        shared_arr[tx, ty] = arr[tx + i * TPB, ty + i * TPB]



def main():
    arr = np.zeros(shape=(8,8), dtype=np.int32)
    arr_gpu = cp.zeros(shape=(8,8), dtype=cp.int32)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    gpu_memory_test[blockspergrid, threadsperblock](arr)

    print(arr)
    # print(arr_gpu)

if __name__ == "__main__":
    main()