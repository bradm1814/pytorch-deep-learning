## 00. pytorch fundamentals
import torch
import random

# creating tensors

#scalar
scalar = torch.tensor(7)
print(scalar)

print(scalar.ndim)

print(scalar.item())

#vector
vector = torch.tensor([7,7])

print(vector.ndim)

print(vector.shape)

#MATRIX

MATRIX = torch.tensor([[7,7],
                       [7,8]])
print(MATRIX.ndim)
print(MATRIX.shape)

#Tensor

TENSOR = torch.tensor([[[1,2,3],
                         [3,6,9],
                         [4,5,6]]])

print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])

#my tensor

MY_TENSOR = torch.tensor([[[1,2,3,4],
                            [4,5,6,7],
                            [1,5,6,7],
                            [1,2,3,4]]])

print(MY_TENSOR[0])
print(MY_TENSOR.shape)

#Random tensors

#create a random tensor of size(3,4)

random_tensor = torch.rand(1,3,4)

print(random_tensor)

#create a random tensor with similar shape to an image tensor

random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height , width, color channels RGB

print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

# create 0's and 1's tensors

# create a tensor of all zeros

zero = torch.zeros(size=(3,4))
print(zero)

#create a tensor of all ones

ones = torch.ones(size=(3,4))
print(ones)

#create a range of tensors and tensors-like

# use torch.range()
range_tensor = torch.arange(start = 0, end = 1000, step=77)
print(range_tensor)

#creating tensors like
tensor_like = torch.zeros_like(input=range_tensor)
print(tensor_like)

#tensor data types

# float32 tensor
float_32_tensor = torch.tensor([3,6,9],
                               dtype=None, #datatype is the tensor type
                               device="cuda", # this decides what device handles the calculation
                               requires_grad=False)#whether or not to track gradients with this tensors operations


print(float_32_tensor)

float_16_tensor = float_32_tensor.type(torch.float16)

int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32)

print(int_32_tensor)

print(float_32_tensor*float_16_tensor)

#getting information from tensors

#tensor.dtype gives datatype
#tensor.shape gives tensor shape
#tensor.device get dive info from tensor

print(float_16_tensor.device)

# manipulating tensor (Tensor Operations)

# tensor operations include:
#addition
#subtraction
#multiplication (elemnt-wise)
#division
#matrix multiplication

#create a tensor

tensor = torch.tensor([1,2,3])
print(tensor+10)
print(tensor*10)
print(tensor-10)


print(tensor.device)

# tryout Pytorch built in functions
torch.mul(tensor,10)
torch.add(tensor, 10)

