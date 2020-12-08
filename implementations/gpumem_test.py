TPB = 4

from numba import cuda, int32
import numpa as np
import cupy as cp

@cuda.jit
def gpu_memory_test(arr):
    x, y = cuda.grid(2)
    width, height = arr.shape
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    shared_h = cuda.shared.array(shape=(TPB, TPB), dtype=int32)

    for i in range(bpg):
        shared_h[tx, ty] = h[tx + i * TPB, ty + i * TPB]



def main():
    arr = np.zeros(shape=(8,8), dtype=np.int32)
    arr_gpu = cp.zeros(shape=(8,8), dtype)

    print(arr)
    print(arr_gpu)
    
if __name__ == "__main__":
    main()