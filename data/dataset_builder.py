

import torch
import torch.utils
import torch.utils.data
import os
import sys
from data.sast.java_depreaction.ast_api import create_ast
from util.setting import log
from data.graph_builder.ast_builder import ast_type_dict_generator, gen_ast_graph
from data.data_tool import split_data, split_data_2
from data.sast.java_2.ast_api import create_ast as create_ast_2
from data.sast.java_2.ast_api import createseparategraph
from data.graph_builder.graph_builder import build_graph, build_graph_visualization
import time
class Dataset:
    def __init__(self, dataset_params):
        self.train_data = []
        self.test_data = []
        self.val_data = []
        self.vocab_length = 0 


@DeprecationWarning
class BigCloneBenchDataset(Dataset):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        data_representation = dataset_params['data_representation']
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


class FAJavaDataset(Dataset):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        files_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_source/dataset_java_small/')
        dataset = dataset_params['name']
        graphmode = dataset_params['graph_model']
        nextsib = dataset_params['nextsib']
        ifedge = dataset_params['ifedge']
        whileedge = dataset_params['whileedge']
        foredge = dataset_params['foredge']
        blockedge = dataset_params['blockedge']
        nexttoken = dataset_params['nexttoken']
        nextuse = dataset_params['nextuse']
        if not os.path.exists(files_path):
            log.error(f"dataset not found!! in {files_path}")
            exit(-1)
        else:
            log.info("Creating AST...")
            astdict, vocabsize, vocabdict = create_ast_2(os.path.join(files_path, 'dataset_files'))

        log.info("Creating separate graph...")
        treedict=createseparategraph(astdict, vocabdict,mode=graphmode,nextsib=nextsib,ifedge=ifedge,whileedge=whileedge,foredge=foredge,blockedge=blockedge,nexttoken=nexttoken,nextuse=nextuse)

        log.info("Splitting data...")
        labels_path = os.path.join(files_path, 'clone_labels.txt')
        if not os.path.exists(labels_path):
            log.error(f"labels not found!! in {labels_path}")
            exit(-1)
        else:
            with open(labels_path, 'r') as f:
                labels = f.readlines()
                labels = [label.strip().split(',') for label in labels]
                self.train_data, self.test_data, self.val_data = split_data_2(treedict, labels, dataset)

        log.info("Dataset loaded successfully")
        self.vocab_length = vocabsize




class SemanticCodeGraphJavaDataset(Dataset):
    def __init__(self, dataset_params):
        super().__init__(dataset_params)
        files_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_source/dataset_java/')
        labels_path = os.path.join(files_path, 'clone_labels.txt')
        visualization = dataset_params['visualization']
        dataset = dataset_params['name']
        self.ast_edge = dataset_params['ast_edge']
        self.value_edge = dataset_params['value_edge']
        self.cfg_edge = dataset_params['cfg_edge']
        self.dfg_edge = dataset_params['dfg_edge']
        self.if_augument = dataset_params['if_augument']
        self.loops_augument = dataset_params['loops_augument']
        

        if not os.path.exists(files_path):
            log.error(f"dataset not found!! in {files_path}")
            exit(-1)
        else:
            log.info("Creating AST...")
            astdict, vocabsize, vocabdict, filtered_labels = create_ast_2(os.path.join(files_path, 'dataset_files'), labels_path, dataset)
        
        log.info("Creating separate graph...")
        if visualization:
            key = '/home/zixian/PycharmProjects/semantic_graph_code_code_clone/data/data_source/dataset_java_small/dataset_files/1362.java'
            astdict =  {key: astdict[key]} if key in astdict else {}

   
            newtree, edgesrc, edgetgt, edge_attr = build_graph_visualization(astdict, vocabdict, self.ast_edge, self.value_edge, self.cfg_edge, self.dfg_edge, self.if_augument, self.loops_augument)


            self.newtree, self.edgesrc, self.edgetgt, self.edge_attr = newtree, edgesrc, edgetgt, edge_attr
            return

        start_time = time.time()
        graph_dict = build_graph(astdict, vocabdict, self.ast_edge, self.value_edge, self.cfg_edge, self.dfg_edge, self.if_augument, self.loops_augument)

        end_time = time.time()
        log.info(f"Time taken to build visualization: {end_time - start_time}")


        log.info("Splitting data...")
        if not os.path.exists(labels_path):
            log.error(f"labels not found!! in {labels_path}")
            exit(-1)
        else:
            self.train_data, self.test_data, self.val_data = split_data_2(graph_dict, filtered_labels, dataset)

        log.info("Dataset loaded successfully")
        self.vocab_length = vocabsize
        self.graph_dict = graph_dict

        
        
    def get_visualization(self):
        return self.newtree, self.edgesrc, self.edgetgt, self.edge_attr