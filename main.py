import cerebras.pytorch as cstorch
import torch
from torch.nn import Module
from torch.linalg import cholesky

cstorch.backend(backend_type="CSX")

class Model(Module):
    def __init__(self, A, B):
        super(Model, self).__init__()
        self.A = A
        self.B = B

    def forward(self, x):
        return self.A + self.B

model = Model(torch.rand(5,5), torch.rand(5,5))

print(model)
print(model.A)
print(model.B)
print(model(2))
