

import torch
import torch.utils
import torch.utils.data
import os
import sys
from data.sast.java.ast_api import create_ast
from util.setting import log
from data.graph_builder.ast_builder import ast_type_dict_generator, gen_ast_graph
from data.data_tool import split_data

class Dataset:
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.vocab_length = 0 


class BigCloneBenchDataset(Dataset):
    def __init__(self, data_representation='ast-type-only'):
        super().__init__()
        files_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_source/bigclonebench/')
        if not os.path.exists(files_path):
            log.error("BigCloneBench dataset not found!!")
            exit(-1)
        else:
            ast_list = create_ast(os.path.join(files_path, 'bigclonebench'))
        
        
        if data_representation == 'ast-type-only':
            type_dict, type_length = ast_type_dict_generator(ast_list)
            graph_dict = gen_ast_graph(ast_list, type_dict)
            # train_data, test_data, val_data = self._split_data(ast_list)

        labels_path = os.path.join(files_path, 'clone_labels.txt')
        if not os.path.exists(labels_path):
            log.error("BigCloneBench labels not found!!")
            exit(-1)
        else:
            with open(labels_path, 'r') as f:
                labels = f.readlines()
                labels = [label.strip().split(',') for label in labels]
                self.train_data, self.test_data, self.val_data = split_data(graph_dict, labels)

        self.vocab_length = type_length


            