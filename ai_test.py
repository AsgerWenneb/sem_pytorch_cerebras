# Import the Cerebras PyTorch module
import cerebras.pytorch as cstorch
import torch
import torch.nn.functional as F

from typing import Literal
from cerebras.modelzoo.config import ModelConfig


class MyMLPConfig(ModelConfig):
    name: Literal["my_mlp"]        # <- required


# Define a model
class Model(torch.nn.Module):
    def __init__(self, config: MyMLPConfig):
        if isinstance(config, dict):
            config = MyMLPConfig(**config)

        super().__init__()

        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, data):
        inputs, labels = data
        inputs = torch.flatten(inputs, 1)
        outputs = F.relu(self.fc1(inputs))
        pred_logits = F.relu(self.fc2(outputs))

        loss = torch.nn.NLLLoss()(pred_logits, labels)

        return loss


def build_model(cfg):              # cfg is MyMLPConfig
    return Model()


# Define a data loader
def get_torch_dataloader(batch_size, train):
    from torchvision import datasets, transforms

    train_dataset = datasets.MNIST(
        "./data/mnist",  # change this path
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        target_transform=transforms.Lambda(
            lambda x: torch.as_tensor(x, dtype=torch.int32)
        ),
    )

    return torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )


def main():
    # Define a model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 256)
            self.fc2 = torch.nn.Linear(256, 10)

        def forward(self, inputs):
            inputs = torch.flatten(inputs, 1)
            outputs = F.relu(self.fc1(inputs))
            return F.relu(self.fc2(outputs))

    backend = cstorch.backend(
        "CPU",
        # cluster_config=cstorch.distributed.ClusterConfig(
        #     max_wgt_servers=1,
        #     max_act_per_csx=1,
        #     num_workers_per_csx=1,
        # ),
    )

    model = Model()

    # Compile the model
    compiled_model = cstorch.compile(model, backend=backend)

    # Define an optimizer
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define a data loader
    def get_torch_dataloader(batch_size, train):
        from torchvision import datasets, transforms

        train_dataset = datasets.MNIST(
            "./data/mnist",  # change this path
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
            target_transform=transforms.Lambda(
                lambda x: torch.as_tensor(x, dtype=torch.int64)
            ),
        )

        return torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

    training_dataloader = cstorch.utils.data.DataLoader(
        get_torch_dataloader, batch_size=64, train=True
    )

    # Define the training step
    loss_fn = torch.nn.CrossEntropyLoss()

    @cstorch.trace
    def training_step(inputs, targets):
        outputs = compiled_model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss

    @cstorch.step_closure
    def print_loss(loss: torch.Tensor, step: int):
        print(f"Train Loss {step}: {loss.item()}")

    @cstorch.checkpoint_closure
    def save_checkpoint(step):
        cstorch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"checkpoint_{step}.mdl",
        )

    global_step = 0

    train_executor = cstorch.utils.data.DataExecutor(
        training_dataloader,
        num_steps=100,
        checkpoint_steps=50,
    )

    model.train()
    for inputs, targets in train_executor:
        loss = training_step(inputs, targets)
        print_loss(loss, global_step)
        global_step += 1
        save_checkpoint(global_step)


if __name__ == "__main__":
    main()
