from data.sast.java_depreaction.fun_unit import FunUnit
from typing import List, Dict
from util.setting import log
from data.graph_builder.code_graph import Code_graph, EDGE_DICT


def get_ast_edge(node, nodeindexlist, vocabdict, src, tgt, edgetype):
    token = node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        if not isinstance(child.data, str):
            src.append(node.id)
            tgt.append(child.id)
            edgetype.append([EDGE_DICT['ast_edge']])
            get_ast_edge(child, nodeindexlist, vocabdict, src, tgt, edgetype)
        else:
            get_ast_edge(child, nodeindexlist, vocabdict, src, tgt, edgetype)
        

def get_value_edge(node, src, tgt, edgetype):
    token = node.token
    for child in node.children:
        if isinstance(child.data, str):
            src.append(node.id)
            tgt.append(child.id)
            edgetype.append([EDGE_DICT['value_edge']])
            get_value_edge(child, src, tgt, edgetype)
        else:
            get_value_edge(child, src, tgt, edgetype)



def ast_type_dict_generator(ast_list: List[FunUnit]) -> dict:
    """Generate ast type dictionary from ast  ast list.
    """
    type_list = []
    type_dict = dict()
    for func in ast_list:
        type_list += list(set(func.gen_type_sequence()))

    type_list = list(set(type_list))
    type_length = len(type_list)

    for idx in range(type_length):
        type_dict[type_list[idx]] = idx
    
    return type_dict, type_length




def gen_ast_graph(ast_list: List[FunUnit], type_dict: Dict) -> Dict:
    """Generate ast edge list from ast list.
    """
    output = dict()
    for func in ast_list:
        x = func.gen_identifer_token_sequence()
        edge = func.gen_ast_edge()
        file_name = func.file_name


        x_values = []
        for key, value in x.items():
            if value not in type_dict:
                log.error(f"Type {value} not in type_dict")
                exit(-1)
        x = {k: type_dict[v] for k, v in x.items()}
        
        indexed_x = {k: index for index, k in enumerate(x)}
        output_x = list(x.values())

        edge[0] =  [indexed_x[key] for key in edge[0] if key in indexed_x]
        edge[1] =  [indexed_x[key] for key in edge[1] if key in indexed_x]
        output[file_name] = [output_x, edge]
    return output




    
