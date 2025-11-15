## 00. pytorch fundamentals
import torch
import random

# # creating tensors

# #scalar
# scalar = torch.tensor(7)
# print(scalar)

# print(scalar.ndim)

# print(scalar.item())

# #vector
# vector = torch.tensor([7,7])

# print(vector.ndim)

# print(vector.shape)

# #MATRIX

# MATRIX = torch.tensor([[7,7],
#                        [7,8]])
# print(MATRIX.ndim)
# print(MATRIX.shape)

# #Tensor

# TENSOR = torch.tensor([[[1,2,3],
#                          [3,6,9],
#                          [4,5,6]]])

# print(TENSOR.ndim)
# print(TENSOR.shape)
# print(TENSOR[0])

# #my tensor

# MY_TENSOR = torch.tensor([[[1,2,3,4],
#                             [4,5,6,7],
#                             [1,5,6,7],
#                             [1,2,3,4]]])

# print(MY_TENSOR[0])
# print(MY_TENSOR.shape)

# #Random tensors

# #create a random tensor of size(3,4)

# random_tensor = torch.rand(1,3,4)

# print(random_tensor)

# #create a random tensor with similar shape to an image tensor

# random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height , width, color channels RGB

# print(random_image_size_tensor.shape)
# print(random_image_size_tensor.ndim)

# # create 0's and 1's tensors

# # create a tensor of all zeros

# zero = torch.zeros(size=(3,4))
# print(zero)

# #create a tensor of all ones

# ones = torch.ones(size=(3,4))
# print(ones)

# #create a range of tensors and tensors-like

# # use torch.range()
# range_tensor = torch.arange(start = 0, end = 1000, step=77)
# print(range_tensor)

# #creating tensors like
# tensor_like = torch.zeros_like(input=range_tensor)
# print(tensor_like)

# #tensor data types

# # float32 tensor
# float_32_tensor = torch.tensor([3,6,9],
#                                dtype=None, #datatype is the tensor type
#                                device="cuda", # this decides what device handles the calculation
#                                requires_grad=False)#whether or not to track gradients with this tensors operations


# print(float_32_tensor)

# float_16_tensor = float_32_tensor.type(torch.float16)

# int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32)

# print(int_32_tensor)

# print(float_32_tensor*float_16_tensor)

# #getting information from tensors

# #tensor.dtype gives datatype
# #tensor.shape gives tensor shape
# #tensor.device get dive info from tensor

# print(float_16_tensor.device)

# # manipulating tensor (Tensor Operations)

# # tensor operations include:
# #addition
# #subtraction
# #multiplication (elemnt-wise)
# #division
# #matrix multiplication

# #create a tensor

# tensor = torch.tensor([1,2,3])
# print(tensor+10)
# print(tensor*10)
# print(tensor-10)


# print(tensor.device)

# # tryout Pytorch built in functions
# torch.mul(tensor,10)
# torch.add(tensor, 10)

# # matrix multiplication: 
# # Two main ways of performing multipliocation in neural networks and deep learning:
# # element wise multiplication
# # Matric Multiplication


# print(tensor, "*", tensor)
# print(f"Equals: {tensor*tensor}")


# print(torch.matmul(tensor, tensor))

# #Shapes for matrix multiplication

# tensor_A = torch.tensor([[1,2],
#                         [3,4],
#                         [5,6]])
# tensor_B = torch.tensor([[7,10],
#                         [8,11],
#                         [9,12]])

#to fix our tensor shjape issues we can manipulate the shape of one of our tensors with transpose

# A transpose switches the axis or dimensions of a given tensor


# print(torch.mm(tensor_A, tensor_B.T))

## Finding min max mean sum etc (Tensor aggregation)

# create a tensory

# x= torch.arange(0, 100, 10)

# print(torch.min(x))
# print(torch.max(x))
# print(torch.mean(x.type(torch.float32)))

#reshaping, stacking, squeeaing and unsqueezing

# reshaping - reshapes an input tensor to a defined shape

# View- Reutrn a view of an input tensor of certain shape but keep the same memory as the orginal tensor

# STacking - Combine multiple tensors on top of each other (vstack) vertical stack or side by side (hstack)

# squeez - removes all '1' dimensions from a tensor

# unsqueeze - addsa  1 dimension to our target tensor

# permute - reutrn a view of the input with dimensions permuted (swapped) in a certain way

# let's create a tensor

# x = torch.arange(1., 10.)
# print(x)
# print(x.shape)

#Add an extra dimension ** dimensions must equal the same amount of elments. torch.arange(1.,10.) creates 9 elements
# x_reshaped = x.reshape(1,9) #multipl of 9
# x_reshapedAgain = x.reshape(9,1) #multiple of 9
# x_anotherone = x.reshape(3,3) #multiple of 9

# print(x_reshaped)
# print(x_reshapedAgain)
# print(x_anotherone)

# z= x.view(1,9)

# #changing z changes x because a view of a tendor shares the same memory as the orginal
# print(z)
# print(z.shape)
# z[:,0] = 5
# print(z)
# print(x)

# #Stack tensors on top of each other
# x_stacked = torch.stack([x,x,x,x], dim=0)
# print(x_stacked)

# x_stacked = torch.stack([x,x,x,x], dim=1)
# print(x_stacked)

# #removes torch.squeeze removes all target tensors with dimnesion 1

# print(f"previous tensor = {x_reshaped}")

# print(f"new tensor = {x_reshaped.squeeze()}")

# x_reshaped = x_reshaped.squeeze()

# # torch.unsqueeze() adds a single dimension to a target tensor at a specigic dimension

# print(f"previous target = {x_reshaped}")

# x_unsqueezed = x_reshaped.unsqueeze(dim=0)

# print(f"after unsqueeze: {x_unsqueezed}")

#torch.permute rearranges the dimension of a target tensor in a specified order

x_original = torch.rand(size=(224,224,3)) # height, width, color channels

#permute the oridingal tensor to rearrange the axis(or dimension) order

x_permuted = x_original.permute(2,0,1)# shifts acis to index postions

print(f"orinigal shape: {x_original.shape}")

print(f"new shape: {x_permuted.shape}")





