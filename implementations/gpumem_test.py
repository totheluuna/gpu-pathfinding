TPB = 4

from numba import cuda, int32
import numpy as np
import cupy as cp
import math
from skimage.util.shape import view_as_windows

dim = (8,8)
@cuda.jit
def gpu_memory_test(arr, block, thread, shared_sum_arr, local_sum_arr, padded_arr):
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
    if x >= width and y >= height:
        return

    # initializing local array
    local_arr = cuda.local.array(dim, int32)
    for i in range(TPB):
        for j in range(TPB):
            local_arr[i,j] = 1
    cuda.syncthreads()

    # initializing shared array
    shared_arr = cuda.shared.array((TPB, TPB), int32)
    shared_arr[tx,ty] = arr[x, y]
    cuda.syncthreads()

    padded_arr[x+1,y+1] = arr[x,y]
    cuda.syncthreads()

    block[x , y] = bx * dim_x + by
    thread[x, y] = tx * TPB + ty
    cuda.syncthreads()

    # initializing constant memory


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

@cuda.jit
def constant_mem_test(constant_sum_arr, chunks):
    x, y = cuda.grid(2)
    width, height = dim

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
def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def main():
    width, height = dim
    arr = np.arange(dim[0]*dim[1]).reshape(dim).astype(np.int32)
    thread = np.zeros(shape=dim, dtype=np.int32)
    block = np.zeros(shape=dim, dtype=np.int32)
    shared_sum_arr = np.zeros(shape=dim, dtype=np.int32)
    local_sum_arr = np.zeros(shape=dim, dtype=np.int32)
    constant_sum_arr = np.zeros(shape=dim, dtype=np.int32)
    padded_arr = np.zeros(shape=(dim[0]+2, dim[1]+2), dtype=np.int32)
    # arr_gpu = cp.zeros(shape=(8,8), dtype=cp.int32)

    block = blockshaped(block, TPB, TPB)
    for i in range(block.shape[0]):
        block[i,:] = i
    block = unblockshaped(block, width, height)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print('blocks per grid: ', blockspergrid, '\nthreads per block: ', threadsperblock)
    gpu_memory_test[blockspergrid, threadsperblock](arr, block, thread, shared_sum_arr, local_sum_arr, padded_arr)

    print('Array: ')
    print(arr)
    print('Blocked Array: ')
    print(blockshaped(arr, 4,4))
    print('Block: ')
    print(block)
    print('Thread: ')
    print(thread)
    print('Shared Sum Array: ')
    print(shared_sum_arr)
    print('Local Sum Array: ')
    print(local_sum_arr)
    print(padded_arr)
    chunks = view_as_windows(padded_arr, (TPB+2, TPB+2), step=TPB)
    print(chunks.shape)
    print(chunks)

if __name__ == "__main__":
    main()