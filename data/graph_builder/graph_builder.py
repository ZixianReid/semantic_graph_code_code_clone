import os
from util.setting import log
from javalang.ast import Node
from anytree import AnyNode, RenderTree
from data.graph_builder.ast_builder import get_ast_edge, get_value_edge
from data.graph_builder.cfg_builder import get_cfg_edge
from data.graph_builder.fa_builder import get_if_edge, get_loops_edge
from data.graph_builder.dfg_builder import get_dfg_edge
from data.sast.java_2.ast_api import createtree




def build_graph(astdict,vocabdict, ast_edge, value_edge, cfg_edge, dfg_edge, if_augument, loops_augument):
    graph_dict = {}
    for path,tree in astdict.items():
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None, is_statement=False)
        createtree(newtree, tree, nodelist)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        
        if ast_edge:
            get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
        else:
            log.error("AST edge is compulsory!!!")
            exit(-1)
        
        if value_edge:
            get_value_edge(newtree, edgesrc, edgetgt, edge_attr)

        if cfg_edge:
            try:
                get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)
            except:
                print(path)

        if dfg_edge:
            pass

        if if_augument:
            get_if_edge(newtree, edgesrc, edgetgt, edge_attr)
        
        if loops_augument:
            get_loops_edge(newtree, edgesrc, edgetgt, edge_attr)
            
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
        if ast_edge:
            get_ast_edge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
        else:
            log.error("AST edge is compulsory!!!")
            exit(-1)
        
        if value_edge:
            get_value_edge(newtree, edgesrc, edgetgt, edge_attr)
        
        if cfg_edge:
            pass
            get_cfg_edge(newtree, edgesrc, edgetgt, edge_attr)
        
        if dfg_edge:
            pass

        if if_augument:
            get_if_edge(newtree, edgesrc, edgetgt, edge_attr)
        
        if loops_augument:
            get_loops_edge(newtree, edgesrc, edgetgt, edge_attr)
    
    return newtree, edgesrc, edgetgt, edge_attr