
from data.dataset_builder import Dataset,SemanticCodeGraphJavaDataset, FAJavaDataset
from util.setting import log

# @DeprecationWarning
# def LoadData(DATASET_NAME: str, dataset_params) -> Dataset:
#     if DATASET_NAME == 'BigCloneBench':
#         return BigCloneBenchDataset(dataset_params)
#     elif DATASET_NAME == 'BigCloneBench2':
#         return BigCloneBenchDataset2(dataset_params)

def LoadData(Language:str, dataset_params) -> Dataset:
    if Language == 'Java' and dataset_params['graph_model'] == 'astandnext': 
        return FAJavaDataset(dataset_params)
    elif Language == 'Java' and dataset_params['graph_model'] == 'semantic_code_graph':
        return SemanticCodeGraphJavaDataset(dataset_params)
    else:
        log.error("Invalid language")
        exit(-1)


