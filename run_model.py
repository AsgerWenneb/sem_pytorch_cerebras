from cerebras.modelzoo.common.run_utils import run

from ai_test import (
    Model,
    get_torch_dataloader,
)


def get_train_dataloader(params):
    input_params = params["train_input"]
    batch_size = input_params.get("batch_size")
    get_torch_dataloader(batch_size, True)


def get_eval_dataloader(params):
    input_params = params["eval_input"]
    batch_size = input_params.get("batch_size")
    get_torch_dataloader(batch_size, False)


def main():
    # run(Model, get_train_dataloader, get_eval_dataloader)
    run()


if __name__ == '__main__':
    main()
