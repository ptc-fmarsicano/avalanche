################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Vincenzo Lomonaco, Hamed Hemati                                   #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Naive strategy in an online benchmark
created using OnlineCLScenario.
"""

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.models import SimpleMLP
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch.optim import Adam


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    n_batches = 5
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose(
        [
            RandomCrop(28, padding=4),
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # ---------

    # --- BENCHMARK CREATION
    mnist_train = MNIST(
        root=default_dataset_location("mnist"),
        train=True,
        download=True,
        transform=train_transform,
    )
    mnist_test = MNIST(
        root=default_dataset_location("mnist"),
        train=False,
        download=True,
        transform=test_transform,
    )
    benchmark = nc_benchmark(
        mnist_train, mnist_test, n_batches, task_labels=False, seed=1234
    )
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CREATE THE STRATEGY INSTANCE (ONLINE-NAIVE)
    cl_strategy = OnlineNaive(
        model,
        Adam(model.parameters(), lr=0.1),
        CrossEntropyLoss(),
        train_passes=1,
        train_mb_size=10,
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    # Create online benchmark
    batch_streams = benchmark.streams.values()
    # ocl_benchmark = OnlineCLScenario(batch_streams)
    for i, exp in enumerate(benchmark.train_stream):
        # Create online scenario from experience exp
        ocl_benchmark = OnlineCLScenario(original_streams=batch_streams,
                                         experiences=exp,
                                         experience_size=10,
                                         access_task_boundaries=True)

        # Train on the online train stream of the scenario
        cl_strategy.train(ocl_benchmark.train_stream)

        results.append(cl_strategy.eval(ocl_benchmark.original_test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
