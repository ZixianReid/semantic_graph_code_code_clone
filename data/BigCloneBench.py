

import torch
import torch.utils
import torch.utils.data
import os
import sys
from data.sast.java.ast_api import create_ast
from util.setting import log

class BigCloneBenchDataset(torch.utils.data.Dataset):
    def __init__(self, data_representation='ast-type-only'):
        files_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_source/bigclonebench/bigclonebench')
        if not os.path.exists(files_path):
            log.error("BigCloneBench dataset not found!!")
            exit(-1)
        else:
            ast_list = create_ast(files_path)
        # data = []
        # self.train = data['train']
        # self.test = data['test']
        # self.val = data['val']

        # self.name = 'BigCloneBench'


