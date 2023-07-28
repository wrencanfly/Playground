import torch
import numpy as np

# initialize a tensor for given data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# from anothor tensor
x_ones = torch.ones_like(x_data)
x_rand = troch.rand_like(x_data)

# from constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# attriutes
tensor = torch.rand(2,3)
tensor_shape = tensor.shape
tensor_dtype = tensor.dtype
tensor_device = tensor.device

# move the tensor to the GPU
if tensor.cuda.is_available():
    tensor = tensor.to("cuda")

# standard numpy-like indexing and slicing operations
tensor = torch.ones(4,4)
first_row = tesnor[0]
first_col = tensor[:,0]
last_col = tensor[...,-1] # use ellipsis to make it clean
tensor[:,1] = 0 # this will make the second column 0

# join tensors
t1 = torch.cat([tensor, tensor, tensor], dim = 1) 
# dim = 0 -> stack on rows, while dim = 1 -> stack on columns

# arithmetic operations
# -> matrix multiplication
y1 = tensor @ tensor.T # @ sign represents matrix multiplication
y2 = torch.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

# -> element-wise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

# -> single-element tensors
agg = tensor.sum()
agg_item = agg.item() # convert it to a python numerical value, type(agg_item) is 'float'

# -> inplace operations, underscore
tensor.add_(5)




