TPB = 4

from numba import cuda, int32
import numpy as np
import cupy as cp
import math

dim = (8,8)
@cuda.jit
def gpu_memory_test(arr):
    x, y = cuda.grid(2)
    width, height = dim
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    dim_x = cuda.blockDim.x
    dim_y = cuda.blockDim.y
    bpg_x = cuda.gridDim.x
    bpg_y = cuda.gridDim.y
    bpg = bpg_x

    # print(bpg)
    if x >= arr.shape[0] and y >= arr.shape[1]:
        return

    local_arr = cuda.local.array(dim, int32)
    for i in range(TPB):
        for j in range(TPB):
            local_arr[i,j] = 1
    # local_arr[x,y] = 1
    cuda.syncthreads()
    # shared_arr[tx,ty] = arr[tx, ty]
    # cuda.syncthreads()
    # arr[tx,ty] = shared_arr[tx, ty]*2
    # cuda.syncthreads()

    # arr[x , y] = bx * dim_x + by
    sum = 0
    for i in range(width):
        for j in range(height):
            sum += local_arr[i,j]
    arr[x,y] = sum
    cuda.syncthreads()

def main():
    arr = np.zeros(shape=dim, dtype=np.int32)
    # arr_gpu = cp.zeros(shape=(8,8), dtype=cp.int32)

    w, h = arr.shape
    for i in range(w):
        for j in range(h):
            arr[i,j] = i * w + j

    print(arr)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print('blocks per grid: ', blockspergrid, '\nthreads per block: ', threadsperblock)
    gpu_memory_test[blockspergrid, threadsperblock](arr)

    print(arr)
    # print(arr_gpu)

if __name__ == "__main__":
    main()