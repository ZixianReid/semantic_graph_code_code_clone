

class Code_graph:
    def __init__(self, x ,edge, file_name):
        self.x = x
        self.edge = edge
        self.file_name = file_name


class Queue():
    """Construct the queue structure using list
    """
    def __init__(self) -> None:
        self.__list = list()
    
    def is_empty(self):
        return self.__list == []
    
    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.is_empty():
            return False
        
        return self.__list.pop(0)

class Sample:
    """
    A class used to represent a Sample in a dataset.
    Attributes
    ----------
    x1 : Any
        The first code pair feature matrix.
    x2 : Any
        The second code pair feature matrix.
    edge_index_1 : Any
        The edge indices for the first graph.
    edge_index_2 : Any
        The edge indices for the second graph.
    edge_attr_1 : Any
        The edge attributes for the first graph.
    edge_attr_2 : Any
        The edge attributes for the second graph.
    clone_label : Any
        The label indicating whether the graphs are clones.
    dataset_label : Any
        The label of the dataset.
    clone_type : Any
        The type of clone. From T1  to T4
    similarity_score : Any
        The syntactic similarity score between the code paris.
    """
        # Initialization code here

    def __init__(self, x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2, clone_label, dataset_label, clone_type, similarity_score):
        self.x1 = x1
        self.x2 = x2
        self.edge_index_1 = edge_index_1
        self.edge_index_2 = edge_index_2
        self.edge_attr_1 = edge_attr_1
        self.edge_attr_2 = edge_attr_2
        self.clone_label = clone_label
        self.dataset_label = dataset_label
        self.clone_type = clone_type
        self.similarity_score = similarity_score

EDGE_DICT = {"ast_edge":[0], 'value_edge':[1], 'cfg_edge':[2], 'dfg_edge':[3], 'if_edge':[4], 'ifelse_edge': [5], 'while_edge':[6],
             'for_edge':[7]}