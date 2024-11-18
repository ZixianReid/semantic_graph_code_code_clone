from data.graph_builder.code_graph import EDGE_DICT


def get_if_edge(node, src, tgt, edgetype):
    token = node.token
    if token == 'IfStatement':
        src.append(node.children[0].id)
        tgt.append(node.children[1].id)
        edgetype.append([EDGE_DICT['if_edge']])
        src.append(node.children[1].id)
        tgt.append(node.children[0].id)
        edgetype.append([EDGE_DICT['if_edge']])
        if len(node.children) == 3:
            src.append(node.children[0].id)
            tgt.append(node.children[2].id)
            edgetype.append([EDGE_DICT['ifelse_edge']])
            src.append(node.children[2].id)
            tgt.append(node.children[0].id)
            edgetype.append([EDGE_DICT['ifelse_edge']])
    for child in node.children:
        get_if_edge(child, src, tgt, edgetype)


def get_loops_edge(node, src, tgt, edgetype):
    token = node.token
    if token == 'WhileStatement':
        src.append(node.children[0].id)
        tgt.append(node.children[1].id)
        edgetype.append([EDGE_DICT['while_edge']])
        src.append(node.children[1].id)
        tgt.append(node.children[0].id)
        edgetype.append([EDGE_DICT['while_edge']])
    if token == 'ForStatement':
        src.append(node.children[0].id)
        tgt.append(node.children[1].id)
        edgetype.append([EDGE_DICT['for_edge']])
        src.append(node.children[1].id)
        tgt.append(node.children[0].id)
        edgetype.append([EDGE_DICT['for_edge']])
    for child in node.children:
        get_loops_edge(child, src, tgt, edgetype)

