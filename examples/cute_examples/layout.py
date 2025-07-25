import torch
import cutlass
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}") # compime time
    cute.printf("tensor : {} ",tensor) # runtime

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)


@cute.jit
def foo1(tensor):
    print(f"tensor.layout: {tensor.layout}") # compime time
    print(f"cute.size(tensor): {cute.size(tensor)}")
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo1, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different


@cute.jit
def foo2(tensor, x: cutlass.Constexpr[int]):
    print(f"cute.size(tensor): {cute.size(tensor)}")  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo2(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo2(b, 3)                # Second call with dynamic layout

@cute.jit
def print_tensor_basic(x : cute.Tensor):
    # Print the tensor
    print("Basic output:")
    cute.print_tensor(x)

#x = torch.randn(30, 20, device="cpu")
x = torch.randn(30,20, device="cuda")
y = from_dlpack(x)

print(f"y.shape: {y.shape}")        # (30, 20)
print(f"y.stride: {y.stride}")       # (20, 1)
print(f"y.memspace: {y.memspace}")     # generic (if torch tensor in on device memory, memspace will be gmem), if data is on cpu it is 0, if data is on cuda it is 1
print(f"y.element_type: {y.element_type}") # Float32
print(f"y: {y}")              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>
#cute.print_tensor(y)
#print_tensor_basic(y)
# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
# b.dim_order() is (3,2,4,0,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)


# auto deduce the stride order to be [2,1,0,3]
t0 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0, divisibility=2
)
# (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)
print(t0)
#cute.print_tensor(t0)
t1 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
)
# (8,?{div=2},16,2):(2,16,?{div=32},1)
print(t1)

t2 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
).mark_compact_shape_dynamic(
    mode=3, divisibility=2
)
# (8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)
print(t2)

t3 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(3, 0, 2, 4, 1)
)
# (1,4,?,32,1):(0,1,4,?{div=4},0)
print(t3)

t4 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(2, 3, 4, 0, 1)
)
# (1,4,?,32,1):(0,1,128,4,0)
print(t4)

#t5 = t2.mark_compact_shape_dynamic(
#    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
#)
## The stride_order is not consistent with the last stride_order
#
#t6 = from_dlpack(a).mark_compact_shape_dynamic(
#    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
#)
## The stride_order is not consistent with the deduced stride_order
#
#t7 = from_dlpack(b).mark_compact_shape_dynamic(
#    mode=0, divisibility=4
#)
## The layout could not be deduced, please specify the stride_order explicitly
#
#t8 = from_dlpack(b).mark_compact_shape_dynamic(
#    mode=30, divisibility=5, stride_order=(3, 0, 2, 4, 1)
#)
## Expected mode value to be in range [0, 5), but got 30
#
#t9 = from_dlpack(b).mark_compact_shape_dynamic(
#    mode=3, divisibility=5, stride_order=(2, 1, 2, 3, 4)
#)
## Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain 0.
#
#t10 = from_dlpack(b).mark_compact_shape_dynamic(
#    mode=3, divisibility=5, stride_order=(0, 1, 2, 3, 4, 5)
#)
## Expected stride_order to have 5 elements, but got 6.
#
#t11 = from_dlpack(b).mark_compact_shape_dynamic(
#    mode=0, divisibility=4, stride_order=b.dim_order()
#)
## The shape(1) of mode(0) is not divisible by the divisibility(4)
#
#t12 = from_dlpack(b).mark_compact_shape_dynamic(
#    mode=0, divisibility=1, stride_order=(2, 1, 3, 0, 4)
#)
## The stride_order is not consistent with the layout



e_hier = cute.E([0, 1])
print(f"e_hier: {e_hier}")
