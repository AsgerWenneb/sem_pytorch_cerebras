import cerebras.pytorch as cstorch
import torch
from torch.nn import Module


class Multigrid(Module):
    def __init__(self, k, omega, A):
        super(Multigrid, self).__init__()
        self.size = A.size()[0]
        self.A = A
        self.smoothing = Jacobi(omega, A)
        self.k = k
        if (k > 0):
            self.coarse = Multigrid(k-1, omega, A)

    def forward(self, u, b):
        print(f"[{self.k}] Solving with initial value {u}")
        print(f"[{self.k}] Solving with rhs vector {b}")
        # Pre-smoothing
        u = self.smoothing(u, b)

        print(f"[{self.k}] After pre-smooth {u}")

        if (self.k > 0):
            print(f"[{self.k}] Going down a level")
            residual = b - torch.matmul(self.A, u)
            print(f"[{self.k}] residual {residual}")
            delta_u = self.coarse(torch.zeros(self.size), residual)
            u += delta_u
            print(f"[{self.k}] After lower level {u}")

        # Post-smoothing
        u = self.smoothing(u, b)
        print(f"[{self.k}] After post-smooth {u}")

        return u


class Jacobi(Module):
    def __init__(self, omega, A):
        super(Jacobi, self).__init__()
        size = A.size()[0]
        self.omega = omega
        self.A_inv_diag = 1/torch.diagonal(A)
        # Smoothing matrix
        self.R = torch.eye(size) - omega * self.A_inv_diag * A

    def forward(self, u, b):
        # Smoothing vector
        g = self.omega*self.A_inv_diag*b

        for i in range(1):
            u = g + torch.matmul(self.R, u)

        return u


def get_torch_dataloader(batch_size, size):
    return torch.utils.data.DataLoader(
        torch.TensorDataset(torch.eye(1)), batch_size=batch_size
    )


@cstorch.trace
def mg_step(model, *inputs):
    return model(*inputs)


@cstorch.step_closure
def print_step(value: torch.Tensor, step: int):
    print(f"Train Loss {step}: {value.item()}")


def main():
    backend = cstorch.backend(backend_type="CPU")

    omega = 2/3
    A = torch.eye(5)
    b = torch.zeros(5)
    model = Multigrid(1, omega, A)

    compiled_model = cstorch.compile(model, backend=backend)

    u = torch.rand(5)
    print("Initial guess", u)
    u = mg_step(compiled_model, u, b)
    print("Estimated solution", u)


if __name__ == "__main__":
    main()
