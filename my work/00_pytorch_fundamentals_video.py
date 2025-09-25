## 00. pytorch fundamentals
import torch

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