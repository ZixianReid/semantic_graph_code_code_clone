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


def get_next_sib_edge(node, edgesrc, edgetgt, edge_attr):
    for i in range(len(node.children) - 1):
        edgesrc.append(node.children[i].id)
        edgetgt.append(node.children[i + 1].id)
        edge_attr.append([EDGE_DICT['if_edge']])
        edgesrc.append(node.children[i + 1].id)
        edgetgt.append(node.children[i].id)
        edge_attr.append([EDGE_DICT['if_edge']])
    for child in node.children:
        get_next_sib_edge(child, edgesrc, edgetgt, edge_attr)   


def get_next_token_edge(node, edgesrc, edgetgt, edge_attr,tokenlist):

    def get_token_list(node, edgetype, tokenlist):
        if len(node.children) == 0:
            tokenlist.append(node.id)
        for child in node.children: 
            get_token_list(child, edgetype, tokenlist)
    get_token_list(node, edge_attr, tokenlist)
    for i in range(len(tokenlist) - 1):
        edgesrc.append(tokenlist[i])
        edgetgt.append(tokenlist[i + 1])
        edge_attr.append([EDGE_DICT['if_edge']])
        edgesrc.append(tokenlist[i + 1])
        edgetgt.append(tokenlist[i])
        edge_attr.append([EDGE_DICT['if_edge']])



def get_next_use_edge(node, edgesrc, edgetgt, edge_attr, variabledict):
    def getvariables(node,edgetype,variabledict):
        token=node.token
        if token=='MemberReference':
            for child in node.children:
                if child.token==node.data.member:
                    variable=child.token
                    variablenode=child
            if not variabledict.__contains__(variable):
                variabledict[variable]=[variablenode.id]
            else:
                variabledict[variable].append(variablenode.id)
        for child in node.children:
            getvariables(child,edgetype,variabledict)
    getvariables(node,edge_attr,variabledict)
    for key in variabledict.keys():
        for i in range(len(variabledict[key])-1):
            edgesrc.append(variabledict[key][i])
            edgetgt.append(variabledict[key][i+1])
            edge_attr.append([EDGE_DICT['if_edge']])
            edgesrc.append(variabledict[key][i+1])
            edgetgt.append(variabledict[key][i])
            edge_attr.append([EDGE_DICT['if_edge']])