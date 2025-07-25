import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    tidx,_, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf(f"hello word from device")

@cute.jit
def hello_world():
    print(f"hello world from host")
    kernel().launch(grid=(1,1,1), # single thread block
            block=(32,1,1), # one warp(32 threads) per thread block

            )
if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    print("Running hello_world()...")
    hello_world()

    print("compile...")
    h = cute.compile(hello_world)

    print("running compile version...")
    h()

