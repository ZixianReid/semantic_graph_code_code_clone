
from data.BigCloneBench import BigCloneBenchDataset

def LoadData(DATASET_NAME):
    if DATASET_NAME == 'BigCloneBench':
        return BigCloneBenchDataset()
