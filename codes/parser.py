import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='ENERGY',
                    help="dataset from ENERGY")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='LSTM',
                    help="model name")
parser.add_argument('--test',
                    action='store_true',
                    help="test the model")
parser.add_argument('--filter',
                    action='store_true',
                    help="train with filter dataset")
parser.add_argument('--retrain',
                    action='store_true',
                    help="retrain the model")
parser.add_argument('--less',
                    action='store_true',
                    help="train using less data")
args = parser.parse_args()
