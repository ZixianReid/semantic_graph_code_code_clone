
from util.setting import log, init_logging
import json
from train.train_gmn import evaluation_gmn
from nets.load_net import gnn_model
from data.data import LoadData
import torch
from graphviz import Digraph
from anytree import Node, RenderTree


init_logging(20, 'test_gmn.log', '/home/zixian/PycharmProjects/semantic_graph_code_code_clone/out/type_ast_gmn_bcb' )


config = '/home/zixian/PycharmProjects/semantic_graph_code_code_clone/configs/semantic_code_graph_gmm_bcb_visual.json'

with open(config) as f:
    config = json.load(f)


MODEL_NAME = config['model']
params = config['params']


net_params = config['net_params']
net_params['device'] = 'cuda:0'
LANGUAGE = config['language']
dataset_params = config['dataset_params']
dataset = LoadData(LANGUAGE, dataset_params)

newtree, edgesrc, edgetgt, edge_attr = dataset.get_visualization()

def extract_node_edges(root):
    node_dict = {}
    for _, _, node in RenderTree(root):
        node_dict[node.id] = node
    return node_dict


def add_nodes_edges(graph, node_dict, edgesrc, edgetgt, edge_attr):

    color_map = {
        'AST_edge': 'black',
        'CFG_edge': 'green',
        'DFG_edge': 'pink',
        'FA_edge': 'red',
    }

    edge_type_map = {
        0: 'black',
        1: 'black',
        2: 'green',
        3: 'pink',
        4: 'red',
        5: 'red',
        6: 'red',
        7: 'red'
    }

    for node_id, node_obj in node_dict.items():


        graph.node(str(node_id), label=node_obj.token if node_obj.token else 'Unknown')
        # graph.node(str(node_id), label=node_obj.token + "-" + str(node_id)  if node_obj.token else 'Unknown')


    for i in range(len(edgesrc)):
        scr_node = node_dict.get(edgesrc[i])
        tgt_node = node_dict.get(edgetgt[i])
        if scr_node and tgt_node:
            edge_color = edge_type_map[edge_attr[i][0][0]]
            graph.edge(str(scr_node.id), str(tgt_node.id), color=edge_color)
    
    with graph.subgraph(name="legend") as s:
        s.attr(label='Legend', style='dashed')
        for edge_type_name, edge_color in color_map.items():
            s.node(edge_type_name, label=edge_type_name, shape='plaintext', fontcolor=edge_color, fontsize="20")  


g = Digraph('AST', filename='ast_tree.gv', format='png', node_attr={'shape': 'box', 'fontname': 'Arial'})

add_nodes_edges(g, extract_node_edges(newtree), edgesrc, edgetgt, edge_attr)

g.render()

from IPython.display import Image
# Image(filename='ast_tree.gv11.png')



# for pre, fill, node in RenderTree(newtree):
#     print("%s%s" % (pre, node.token + "--" + type(node.data).__name__))


