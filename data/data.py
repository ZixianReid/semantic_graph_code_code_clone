
from data.dataset_builder import Dataset, BigCloneBenchDataset


def LoadData(DATASET_NAME: str) -> Dataset:
    if DATASET_NAME == 'BigCloneBench':
        return BigCloneBenchDataset()



