TPB = 4

from numba import cuda, int32
import numpy as np
import cupy as cp
import math

dim = (8,8)
@cuda.jit
def gpu_memory_test(block, thread, shared_sum_arr, local_sum_arr):
    x, y = cuda.grid(2)
    width, height = block.shape
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
    if x >= width and y >= height:
        return

    # initializing local array
    local_arr = cuda.local.array(dim, int32)
    for i in range(TPB):
        for j in range(TPB):
            local_arr[i,j] = 1
    cuda.syncthreads()

    # initializing shared array
    shared_arr = cuda.shared.array((TPB,TPB), int32)
    shared_arr[tx,ty] = arr[x, y]
    cuda.syncthreads()

    # arr[tx,ty] = shared_arr[tx, ty]*2
    # cuda.syncthreads()

    block[x , y] = bx * dim_x + by
    thread[x, y] = tx * TPB + ty
    cuda.syncthreads()

    shared_sum = 0
    local_sum = 0
    for i in range(TPB):
        for j in range(TPB):
            local_sum += local_arr[i,j]
            shared_sum += shared_arr[i,j]
    # print('running thread: ', tx, ty)
    # print('grid coordinates: ', x, y)
    cuda.syncthreads()
    
    local_sum_arr[x,y] = local_sum
    shared_sum_arr[x,y] = shared_sum
    cuda.syncthreads()
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
def main():
    arr = np.zeros(shape=dim, dtype=np.int32)
    thread = np.zeros(shape=dim, dtype=np.int32)
    block = np.zeros(shape=dim, dtype=np.int32)
    # arr_gpu = cp.zeros(shape=(8,8), dtype=cp.int32)

    w, h = arr.shape
    for i in range(w):
        for j in range(h):
            arr[i,j] = i * w + j

    print(blockshaped(arr, 4,4))


    print(arr)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print('blocks per grid: ', blockspergrid, '\nthreads per block: ', threadsperblock)
    gpu_memory_test[blockspergrid, threadsperblock](block, thread)

    # print(arr)
    # print(arr_gpu)

    print(block)
    print(thread)

if __name__ == "__main__":
    main()