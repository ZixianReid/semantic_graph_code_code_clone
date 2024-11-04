

import argparse
import json
from data.data import LoadData
from util.setting import init_logging, log


def train_val_pipeline():
    pass

def main():
    parser = argparse.ArgumentParser(description='code clone detection')
    parser.add_argument('-l', '--logging', type=int, default=20,
                        help='Log level [10-50] (default: 10 - Debug)')
    parser.add_argument('--config')
    parser.add_argument('--language')
    parser.add_argument('--dataset', default='BigCloneBench')
    parser.add_argument('--model')
    parser.add_argument('--batch_size')
    parser.add_argument('--report', type=str, default="",
                        help='file name to report training logs.') 

    args = parser.parse_args()

    # with open(args.config) as f:
    #     config = json.load(f)
    

    # initial logger
    init_logging(args.logging, args.report)

    # dataset
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        pass
    
    dataset = LoadData(DATASET_NAME)

    
    
    train_val_pipeline()


main()