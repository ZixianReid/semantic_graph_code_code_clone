import os
from util.setting import log
from javalang.ast import Node
from anytree import AnyNode, RenderTree
from data.graph_builder.ast_builder import get_ast_edge, get_value_edge, get_ast_edge_without_value_edge
from data.graph_builder.cfg_builder import get_cfg_edge
from data.graph_builder.fa_builder import get_if_edge, get_loops_edge
from data.graph_builder.dfg_builder import get_dfg_edge
from data.sast.java_2.ast_api import createtree
from data.graph_builder.code_graph import EDGE_DICT
from tqdm import tqdm



def build_graph(astdict,vocabdict, ast_edge, value_edge, cfg_edge, dfg_edge, if_augument, loops_augument):
    graph_dict = {}
    for path,tree in tqdm(astdict.items()):
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None, is_statement=False)
        createtree(newtree, tree, nodelist)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]

        if ast_edge and value_edge:
            get_ast_edge_without_value_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
            get_value_edge(newtree, edgesrc, edgetgt, edge_attr)
        elif ast_edge and not value_edge:
            get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
        else:
            log.error("AST edge is compulsory!!!")
            exit(-1)

        # build cfg edge
        if cfg_edge or dfg_edge:
            get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)

            newtree, edgesrc, edgetgt, edge_attr =  get_dfg_edge(newtree, edgesrc, edgetgt, edge_attr)

            # delete unrequired edges
            if not cfg_edge:
                edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['cfg_edge']])

            if not dfg_edge:
                edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['dfg_edge']])

        if  if_augument:
            get_if_edge(newtree, edgesrc, edgetgt, edge_attr)

        if loops_augument:
            get_loops_edge(newtree, edgesrc, edgetgt, edge_attr)

        edgesrc, edgetgt, edge_attr = remove_duplicates_edges(edgesrc, edgetgt, edge_attr)
        
        edge_index=[edgesrc, edgetgt]
        astlength=len(x)
        file_name = os.path.splitext(os.path.basename(path))[0]
        edge_attr = [inner[0] for inner in edge_attr]
        graph_dict[file_name]=[[x,edge_index,edge_attr],astlength]
    
    return graph_dict

def build_graph_visualization(astdict,vocabdict, ast_edge, value_edge, cfg_edge, dfg_edge, if_augument, loops_augument):
    if len(astdict) != 1:
        log.error("AST dictionary is not 1!!!")
        exit(-1)
    else:
        path, tree =  next(iter(astdict.items()))
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None, is_statement=False)
        createtree(newtree, tree, nodelist)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]

        if ast_edge and value_edge:
            get_ast_edge_without_value_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
            get_value_edge(newtree, edgesrc, edgetgt, edge_attr)
        elif ast_edge and not value_edge:
            get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
        else:
            log.error("AST edge is compulsory!!!")
            exit(-1)

        # build cfg edge
        get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)

        # build dfg edge
        newtree, edgesrc, edgetgt, edge_attr =  get_dfg_edge(newtree, edgesrc, edgetgt, edge_attr)


        # build if edge
        get_if_edge(newtree, edgesrc, edgetgt, edge_attr)

        # build loops edge
        get_loops_edge(newtree, edgesrc, edgetgt, edge_attr)

        # # delete unrequired edges
        if not cfg_edge:
            edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['cfg_edge']])

        if not dfg_edge:
            edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['dfg_edge']])

        if not if_augument:
            edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['if_edge']])
            edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['ifelse_edge']])

        if not loops_augument:
            edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['while_edge']])
            edgesrc, edgetgt, edge_attr = delete_edge(edgesrc, edgetgt, edge_attr, [EDGE_DICT['for_edge']])

        edgesrc, edgetgt, edge_attr = remove_duplicates_edges(edgesrc, edgetgt, edge_attr)

    return newtree, edgesrc, edgetgt, edge_attr



# def build_graph_visualization(astdict,vocabdict, ast_edge, value_edge, cfg_edge, dfg_edge, if_augument, loops_augument):
#     if len(astdict) != 1:
#         log.error("AST dictionary is not 1!!!")
#         exit(-1)
#     else:
#         path, tree =  next(iter(astdict.items()))
#         nodelist = []
#         newtree=AnyNode(id=0,token=None,data=None, is_statement=False)
#         createtree(newtree, tree, nodelist)
#         x = []
#         edgesrc = []
#         edgetgt = []
#         edge_attr=[]
#         if ast_edge:
#             get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
#         else:
#             log.error("AST edge is compulsory!!!")
#             exit(-1)
        
#         if value_edge:
#             get_value_edge(newtree, edgesrc, edgetgt, edge_attr)
        
#         if cfg_edge:
#             get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)
        
#         if dfg_edge:
#            newtree, edgesrc, edgetgt, edge_attr =  get_dfg_edge(newtree, edgesrc, edgetgt, edge_attr)

#         if if_augument:
#             get_if_edge(newtree, edgesrc, edgetgt, edge_attr)
        
#         if loops_augument:
#             get_loops_edge(newtree, edgesrc, edgetgt, edge_attr)
    
#     return newtree, edgesrc, edgetgt, edge_attr

def delete_edge(edgesrc, edgetgt, edge_attr, edge_type):
    deleted_elements = [
        (src, tgt, attr)
        for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr)
        if attr == edge_type
    ]
    filtered = [
        (src, tgt, attr)
        for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr)
        if attr != edge_type
    ]

    edgesrc, edgetgt, edge_attr = zip(*filtered) if filtered else ([], [], [])

    return edgesrc, edgetgt, edge_attr


def remove_duplicates_edges(edgesrc, edgetgt, edge_attr):
    """
    Removes duplicate elements from the given edges based on (edgesrc, edgetgt, edge_attr) tuples.
    """
    seen = set()
    unique_edges = []
    for src, tgt, attr in zip(edgesrc, edgetgt, edge_attr):
        attr_value = attr[0][0]
        edge = (src, tgt, attr_value)
        if edge not in seen:
            unique_edges.append((src, tgt, attr))
            seen.add(edge)
    if unique_edges:
        edgesrc, edgetgt, edge_attr = zip(*unique_edges)
    else:
        edgesrc, edgetgt, edge_attr = [], [], []
    return list(edgesrc), list(edgetgt), list(edge_attr)