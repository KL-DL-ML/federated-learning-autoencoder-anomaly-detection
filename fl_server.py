import argparse
from collections import OrderedDict
from pprint import pprint
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import result

import flwr as fl
from flwr.common import Metrics
import numpy as np
import torch
from time import time

from codes.data_loader import *
from codes.model_utils import *
from codes.evaluations.eval_utils import *

# pylint: disable=no-member
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
# pylint: enable=no-member

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=20,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=2,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--model",
    type=str,
    default="AE",
    help="model to train",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="ENERGY",
    help="dataset to train and test",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[
                Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
                BaseException,
            ]
        ],
    ) -> Optional[fl.common.Weights]:
        os.makedirs(f"training_rounds/{args.model}/{args.dataset}", exist_ok=True)
        weights = super().aggregate_fit(server_round, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {server_round} weights...")
            np.savez(f"training_rounds/{args.model}/{args.dataset}/round-{server_round}-weights.npz", *weights)
        return weights


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def main() -> None:
    print(args)
    
    start = time()

    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server")
    
    config = {
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 15,
    }

    # Load evaluation data
    train_loader, test_loader, labels = load_dataset(args.dataset, filter=False)
    model, optimizer, scheduler, _, _ = load_model(args.model, labels.shape[1], config)
    model_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    
    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = SaveModelStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=args.min_sample_size,
        min_eval_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(model, optimizer, scheduler, trainD, testD, trainO, testO, labels),
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.weights_to_parameters(model_weights),
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    history = fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config={"num_rounds": args.rounds},
    )
    print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
    print('Saving Model')
    save_model(model, optimizer, scheduler)

def save_model(model, optimizer, scheduler):
    try:
        folder = f'checkpoints/{args.model}_{args.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save(
            {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, file_path)
    except:
        print('Cannot save model')
    

def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(15),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
        "model": args.model,
        "dataset": args.dataset,
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    model, optimizer, scheduler, trainD, testD, trainO, testO, labels
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        set_weights(model, weights)
        loss, mae, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        lossT, _, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
        
        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
        
        ls = np.mean(lossFinal)
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        
        TP = float(result['TP'])
        TN = float(result['TN'])
        FP = float(result['FP'])
        FN = float(result['FN'])
        
        # cf_matrix = [[TP, FP], [FN, TN]]
        # plot_confusion_matrix_fl('AE_ENERGY', self.cid, np.asarray(cf_matrix))
        
        TPR = round((TP / (TP + FN)), 6)
        FPR = round((FP / (FP + TN)), 6)
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        # pprint(result)
        print("++++++++> MAE: ", ls)
        print("++++++++> MAE: ", np.mean(mae))
        # print("++++++++> Accuracy: ", accuracy)
        return ls, {"accuracy": accuracy}
    return evaluate


if __name__ == "__main__":
    main()