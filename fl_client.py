import argparse
from pprint import pprint
import timeit
from collections import OrderedDict
from time import time

import flwr as fl
import numpy as np
import torch
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

from codes.data_loader import *
from codes.model_utils import *
from codes.evaluations.eval_utils import *
from codes.plotter import *

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

def get_weights(model: torch.nn.ModuleList) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


class Client(fl.client.Client):
    def __init__(self, 
                cid: str, 
                trainset, 
                testset,
                labels,
                model,
                optimizer,
                scheduler,
                epoch,
                accuracy_list) -> None:
        super().__init__()
        self.cid = cid
        self.trainset = trainset
        self.testset = testset
        self.labels = labels
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.accuracy_list = accuracy_list


    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)
    

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")
        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])

        # Set model parameters
        set_weights(self.model, weights)
        
        trainD, testD = next(iter(self.trainset)), next(iter(self.testset))
        trainO, testO = trainD, testD
        if self.model.name in ['USAD', 'AE', 'MAD_GAN']:
            trainD, testD = convert_to_windows(trainD, self.model), convert_to_windows(testD, self.model)
        # Train model
        e = self.epoch + 1
        start = time()
        for e in list(range(self.epoch + 1, self.epoch + epochs + 1)):
            lossT, lr = backprop_fl(e, self.cid, self.model, trainD, trainO, self.optimizer, self.scheduler)
            self.accuracy_list.append((lossT, lr))
        # plot_accuracies(self.accuracy_list, 'AE_ENERGY')
        print('Training time: ' + "{:10.4f}".format(time() - start) + ' s')
        
        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)
        
        trainD, testD = next(iter(self.trainset)), next(iter(self.testset))
        trainO, testO = trainD, testD
        
        if self.model.name == 'AE':
            trainD, testD = convert_to_windows(trainD, self.model), convert_to_windows(testD, self.model)
            
        loss, _ = backprop_fl(0, self.cid, self.model, testD, testO, self.optimizer, self.scheduler, training=False)
        lossT, _ = backprop_fl(0, self.cid, self.model, trainD, trainO, self.optimizer, self.scheduler, training=False)
        
        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(self.labels, axis=1) >= 1) + 0
        
        ls = np.mean(lossFinal)
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        
        # Return the number of evaluation examples and the evaluation result (loss)
        TP = float(result['TP'])
        TN = float(result['TN'])
        FP = float(result['FP'])
        FN = float(result['FN'])
        
        # cf_matrix = [[TP, FP], [FN, TN]]
        # plot_confusion_matrix_fl('AE_ENERGY', self.cid, np.asarray(cf_matrix))
        
        TPR = round((TP / (TP + FN)), 6)
        FPR = round((FP / (FP + TN)), 6)
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print('==========================================================')
        print('Client {}'.format(self.cid))
        print('Acc: %.6f%% \nPrecision: %.6f \nRecall: %.6f \nF1score: %.6f \nTPR: %.6f \nFPR: %.6f'%(accuracy*100, 
                                                                                                      result['precision'], 
                                                                                                      result['recall'], 
                                                                                                      result['f1'], 
                                                                                                      TPR, 
                                                                                                      FPR))
        print('==========================================================')
        # print(f"=========> Client {self.cid} Loss: {ls}")
        # print(f"=========> Client {self.cid} Accuracy: {accuracy}")
        print(f"=========> Client {self.cid} Result: {result}")
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(
            loss=float(ls), num_examples=len(self.testset), metrics=metrics
        )


def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:9090",
        required=False,
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--cid", 
        type=str, 
        required=True, 
        help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ENERGY",
        help="Dataset name (default: ENERGY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="AE",
        help="model to train",
    )
    args = parser.parse_args()

    best_configs = {}
    
    best_configs["AE"] = {
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "num_window": 15,
    }

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # load (local, on-device) dataset
    trainset, testset, labels = client_load_dataset(args.dataset, args.cid)

    # model
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], best_configs[args.model])
    model.to(DEVICE)

    # Start client
    client = Client(args.cid, trainset, testset, labels, model, optimizer, scheduler, epoch, accuracy_list)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()