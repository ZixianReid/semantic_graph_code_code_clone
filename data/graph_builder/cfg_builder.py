from data.graph_builder.code_graph import EDGE_DICT, Queue
from util.setting import log


IF_TYPE = ['IfStatement', 'SwitchStatement']
LOOP_TYPE = ['ForStatement', 'EnhancedForControl', 'WhileStatement', 'DoStatement', 'SwitchStatement']
STATEMENT_TYPE = ['MethodDeclaration', 'StatementExpression', 'LocalVariableDeclaration', 'IfStatement', 'ForStatement', 'EnhancedForControl', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'SwitchStatementCase', 'ReturnStatement', 'BlockStatement', 'BreakStatement', 'ContinueStatement', 'TryStatement', 'TryResource', 'CatchClause', 'SynchronizedStatement', 'ThrowStatement', 'AssertStatement']
BLOCK_TYPE = ['BlockStatement', 'TryStatement'] + IF_TYPE + LOOP_TYPE

def get_cfg_edge(node, src, tgt, edgetype):
    node_type = type(node.data).__name__
    if node_type == 'MethodDeclaration':
        entrynode, fringe  = cfg_methoddeclaration(node, src, tgt, edgetype)
    elif node_type == 'StatementExpression':
        entrynode, fringe  = cfg_statementexpression(node, src, tgt, edgetype)
    elif node_type == 'Statement':
        entrynode, fringe  = cfg_statementexpression(node, src, tgt, edgetype)
    elif node_type == 'LocalVariableDeclaration':
        entrynode, fringe  = cfg_localvariabledeclaration(node, src, tgt, edgetype)
    elif node_type == 'IfStatement':
        entrynode, fringe  = cfg_ifstatement(node, src, tgt, edgetype)
    elif node_type == 'ForStatement':
        entrynode, fringe  = cfg_forstatement(node, src, tgt, edgetype)
    elif node_type == 'EnhancedForControl':
        entrynode, fringe  = cfg_forcontrol(node, src, tgt, edgetype)
    elif node_type == 'WhileStatement':
        entrynode, fringe  = cfg_whilestatement(node, src, tgt, edgetype)
    elif node_type == 'DoStatement':
        entrynode, fringe  = cfg_dowhilestatement(node, src, tgt, edgetype)
    elif node_type == 'SwitchStatement':
        entrynode, fringe  = cfg_switchstatement(node, src, tgt, edgetype)
    elif node_type == 'SwitchStatementCase':
        entrynode, fringe  = cfg_switchstatementcase(node, src, tgt, edgetype)
    elif node_type == 'ReturnStatement':
        entrynode, fringe  = cfg_returnstatement(node, src, tgt, edgetype)
    elif node_type == 'BlockStatement':
        entrynode, fringe  = cfg_blockstatement(node, src, tgt, edgetype)
    elif node_type == 'BreakStatement':
        entrynode, fringe  = cfg_breakstatement(node, src, tgt, edgetype)
    elif node_type == 'ContinueStatement':
        entrynode, fringe  = cfg_continuestatement(node, src, tgt, edgetype)
    elif node_type == 'TryStatement':
        entrynode, fringe  = cfg_trystatement(node, src, tgt, edgetype)
    elif node_type == 'TryResource':
        entrynode, fringe  = cfg_trystatement(node, src, tgt, edgetype)
    elif node_type == 'CatchClause':
        entrynode, fringe  = cfg_catchclause(node, src, tgt, edgetype)
    elif node_type == 'SynchronizedStatement':
        entrynode, fringe  = cfg_synchronizedstatement(node, src, tgt, edgetype)
    elif node_type == 'ThrowStatement':
        entrynode, fringe  = cfg_throwstatement(node, src, tgt, edgetype)
    elif node_type == 'AssertStatement':
        entrynode, fringe  = cfg_assertstatement(node, src, tgt, edgetype)
    elif node_type == 'ClassDeclaration':
        entrynode, fringe  = cfg_methoddeclaration(node, src, tgt, edgetype)
    else:
        entrynode, fringe = None, []
    return entrynode, fringe

def cfg_classdeclaration(node, src, tgt, edgetype):
    node.is_statement = True
    chiidren = list(node.children)
    entry_node = node
    fringe_nodes = [node]
    for child in chiidren:
        if type(child.data).__name__ == 'MethodDeclaration':
            entrynode, fringe = get_cfg_edge(child, src, tgt, edgetype)
            fringe_nodes = fringe
    return [entry_node, fringe_nodes]

def cfg_methoddeclaration(node, src, tgt, edgetype):
    node.is_statement = True
    children = list(node.children)
    entry_node = node
    fringe_nodes = [node]
    if len(children) ==0:
        return [entry_node, fringe_nodes]
    else:
        for child in children:
            _s_entrynode, _s_finge = get_cfg_edge(child, src, tgt, edgetype)
            if _s_entrynode is None:
                continue
            for _entrynode in fringe_nodes:
                src.append(_entrynode.id)
                tgt.append(_s_entrynode.id)
                edgetype.append([EDGE_DICT['cfg_edge']])
            fringe_nodes = _s_finge
    return [entry_node, fringe_nodes]

def cfg_statementexpression(node, src, tgt, edgetype):
    return cfg_singlenode(node, src, tgt, edgetype)


def cfg_localvariabledeclaration(node, src, tgt, edgetype):
    return cfg_singlenode(node, src, tgt, edgetype)


def cfg_ifstatement(node, src, tgt, edgetype):
    node.is_statement = True
    entry_node = node
    fringe_nodes = []
    children = list(node.children)
    if len(children) == 2:
        
        # handle for the corner case
        if type(children[-1].data).__name__ == 'Statement':
            fringe_nodes = [node]
        else:
        # handle single if without else and else if
            then_entrynode, then_frige = get_cfg_edge(children[-1], src, tgt, edgetype)
            src.append(entry_node.id)
            tgt.append(then_entrynode.id)
            edgetype.append([EDGE_DICT['cfg_edge']])
            fringe_nodes = then_frige + [node]

    elif len(children) == 3:
        #handle if else else if
        then_entrynode, then_frige = get_cfg_edge(children[-2], src, tgt, edgetype)
        else_entrynode, else_frige = get_cfg_edge(children[-1], src, tgt, edgetype)
        src.append(entry_node.id)
        tgt.append(then_entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
        src.append(entry_node.id)
        tgt.append(else_entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
        fringe_nodes = then_frige + else_frige
    else:
        log.error("If statement has more than 3 children")
        exit(-1)
    return [entry_node, fringe_nodes]


def cfg_forcontrol(node, src, tgt, edgetype):
    node.is_statement = True
    entry_node = node
    children = list(node.children)

    sub_entrynode, fringe = get_cfg_edge(children[-1], src, tgt, edgetype)
    if sub_entrynode:
        src.append(entry_node.id)
        tgt.append(sub_entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])

    for _fringe in fringe:
        src.append(_fringe.id)
        tgt.append(entry_node.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
    
    return [entry_node, fringe + [node]]

def cfg_forstatement(node, src, tgt, edgetype):
    node.is_statement = True
    entry_node = node
    children = list(node.children)

    sub_entrynode, fringe = get_cfg_edge(children[-1], src, tgt, edgetype)
    if sub_entrynode:
        src.append(entry_node.id)
        tgt.append(sub_entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])

    for _fringe in fringe:
        src.append(_fringe.id)
        tgt.append(entry_node.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
    
    return [entry_node, fringe + [node]]


def cfg_whilestatement(node, src, tgt, edgetype):
    node.is_statement = True
    entry_node = node
    children = list(node.children)

    sub_entrynode, fringe = get_cfg_edge(children[-1], src, tgt, edgetype)

    if sub_entrynode:
        src.append(entry_node.id)
        tgt.append(sub_entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
    
    for _fringe in fringe:
        src.append(_fringe.id)
        tgt.append(entry_node.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
    
    return [entry_node, fringe + [node]]


def cfg_dowhilestatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    children = list(node.children)

    sub_entrynode, fringe = get_cfg_edge(children[-1], src, tgt, edgetype)

    if sub_entrynode:
        src.append(entrynode.id)
        tgt.append(sub_entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
    
    for _fringe in fringe:
        src.append(_fringe.id)
        tgt.append(entrynode.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
        
    return [entrynode, fringe]


def cfg_switchstatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    fringe = [node]

    children = list(node.children)
    for child in children:
        _s_entrynode, _s_fringe = get_cfg_edge(child, src, tgt, edgetype)
        if _s_entrynode is None:
            continue
        else:
            pass
            src.append(entrynode.id)
            tgt.append(_s_entrynode.id)
            edgetype.append([EDGE_DICT['cfg_edge']])

        # for _entrynode in fringe:
        #     src.append(_entrynode.id)
        #     tgt.append(_s_entrynode.id)
        #     edgetype.append([EDGE_DICT['cfg_edge']])
        fringe = _s_fringe
    return [entrynode, fringe]
        
         

def cfg_switchstatementcase(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    fringe = [node]

    children = list(node.children)

    for child in children:
        _s_entrynode, _s_fringe = get_cfg_edge(child, src, tgt, edgetype)
        if _s_entrynode is None:
            continue
        for _entrynode in fringe:
            src.append(_entrynode.id)
            tgt.append(_s_entrynode.id)
            edgetype.append([EDGE_DICT['cfg_edge']])
        fringe = _s_fringe
    return [entrynode, fringe]

def cfg_returnstatement(node, src, tgt, edgetype):
    return cfg_singlenode(node, src, tgt, edgetype)


def cfg_blockstatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    fringe = [node]

    children = list(node.children)
    for child in children:
        _s_entrynode, _s_fringe = get_cfg_edge(child, src, tgt, edgetype)
        if _s_entrynode is None:
            continue
        for _entrynode in fringe:
            src.append(_entrynode.id)
            tgt.append(_s_entrynode.id)
            edgetype.append([EDGE_DICT['cfg_edge']])
        fringe = _s_fringe
    return [entrynode, fringe]



def cfg_breakstatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    fringe = []

    parent_type = ['IfStatement', 'SwitchStatement']
    parent_node = seek_parent(node, parent_type)

    # add edge from parent if statement to the next statement sibling
    if parent_node:
        next_sibling = sek_statement_sibling(parent_node)
        if next_sibling:
            src.append(parent_node.id)
            tgt.append(next_sibling.id)
            edgetype.append([EDGE_DICT['cfg_edge']])
    
    # add edge from break statement to the statmeent sibing of the For parent
    _for_parent = seek_parent(node, LOOP_TYPE)

    if not _for_parent:
        return [entrynode, fringe]
    
    next_statement_node = sek_statement_sibling(_for_parent)
    if not next_statement_node:
        return [entrynode, fringe]
    else:
        src.append(node.id)
        tgt.append(next_statement_node.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
        return [entrynode, fringe]



def cfg_continuestatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    fringe = []

    loop_node = seek_parent(node, LOOP_TYPE)
    if loop_node:
        src.append(node.id)
        tgt.append(loop_node.id)
        edgetype.append([EDGE_DICT['cfg_edge']])
    
    parent_node = seek_parent(node, IF_TYPE)
    if parent_node:
        fringe =  [parent_node]
    return [entrynode, fringe]


def cfg_trystatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    fringe = []

    try_statemetn_chidlren = list(node.children)

    try_block_fringe = []
    try_catch_fringe = []

    if len(try_statemetn_chidlren) == 0:
        return [entrynode, [node]]

    for child in try_statemetn_chidlren:
        _s_entrynode, _s_fringe = get_cfg_edge(child, src, tgt, edgetype)
        if _s_entrynode is None:
            continue
        
        _s_entrynode_type = type(_s_entrynode.data).__name__
        if _s_entrynode_type != 'CatchClause':
            if len(_s_fringe) == 0:
                _s_fringe = seek_valid_fringe(_s_entrynode)
                _s_fringe = [_s_fringe]
            try_catch_fringe += _s_fringe
            try_block_fringe += _s_fringe
            src.append(entrynode.id)
            tgt.append(_s_entrynode.id)
            edgetype.append([EDGE_DICT['cfg_edge']])
            fringe = try_block_fringe
        elif _s_entrynode_type == 'CatchClause':
            for _entrynode in try_block_fringe:
                src.append(_entrynode.id)
                tgt.append(_s_entrynode.id)
                edgetype.append([EDGE_DICT['cfg_edge']])
                try_catch_fringe += _s_fringe
            fringe = try_catch_fringe
    return [entrynode, fringe]
    


def cfg_catchclause(node, src, tgt, edgetype):

    # # connect end node to the try statement siblings
    # try_parent = seek_parent(node, ['TryStatement'])
    # if try_parent:
    #     next_statement = sek_statement_sibling(try_parent)

    node.is_statement = True
    entrynode = node
    children = list(node.children)
    fringe = [node]
    if len(children) == 0:
        return [entrynode, fringe]
    else:
        for child in children:
            _s_entrynode, _s_fringe = get_cfg_edge(child, src, tgt, edgetype)
            if _s_entrynode is None:
                continue
            for _entrynode in fringe:
                src.append(_entrynode.id)
                tgt.append(_s_entrynode.id)
                edgetype.append([EDGE_DICT['cfg_edge']])
            fringe = _s_fringe

    return [entrynode, fringe]
    

def cfg_synchronizedstatement(node, src, tgt, edgetype):
    node.is_statement = True
    entrynode = node
    children = list(node.children)
    fringe = [node]
    if len(children) == 0:
        return [entrynode, fringe]
    else:
        for child in children:
            _s_entrynode, _s_fringe = get_cfg_edge(child, src, tgt, edgetype)
            if _s_entrynode is None:
                continue
            for _entrynode in fringe:
                src.append(_entrynode.id)
                tgt.append(_s_entrynode.id)
                edgetype.append([EDGE_DICT['cfg_edge']])
            fringe = _s_fringe
    return [entrynode, fringe]

def cfg_throwstatement(node, src, tgt, edgetype):
    return cfg_singlenode(node, src, tgt, edgetype)

def cfg_assertstatement(node, src, tgt, edgetype):
    return cfg_singlenode(node, src, tgt, edgetype)

def cfg_singlenode(node, src, tgt, edgetype):
    node.is_statement = True
    entry_node = node
    fringe_nodes = [node]
    return [entry_node, fringe_nodes]


def seek_parent(node, parent_type):
    queue = Queue()
    queue.push(node)

    node_flag = list()
    node_flag.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        parent = current_node.parent
        if parent is None:
            return False
        parent_type_name = type(parent.data).__name__
        if parent_type_name not in parent_type:
            if parent not in node_flag:
                queue.push(parent)
                node_flag.append(parent)
        elif parent_type_name in parent_type:
            return parent
    
    return False

def sek_statement_sibling_d(node):
    """Given a node in ast_cpg, find its next statement sibling.
    """
    queue = Queue()
    # Add the logic to find the next statement sibling here
    queue.push(node)
    node_flag = list()
    node_flag.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        parent = current_node.parent

        children = list(parent.children)
        node_idx = children.index(current_node) + 1
        siblings = children[node_idx:]
        _has_statement = False
        next_statement = None
        for sibling in siblings:
            if type(sibling.data).__name__ in STATEMENT_TYPE:
                _has_statement = True
                next_statement = sibling
                break
        if _has_statement:
            return next_statement
        elif not _has_statement and type(parent.data).__name__ in LOOP_TYPE:
            return parent
        else:
            if parent not in node_flag:
                queue.push(parent)
                node_flag.append(parent)
    return False


def sek_statement_sibling(node):
    """Given a node in ast_cpg, find its next statement sibling.
    """

    # Add the logic to find the next statement sibling here

    node_flag = list()
    node_flag.append(node)


    parent = node.parent

    children = list(parent.children)
    node_idx = children.index(node) + 1
    siblings = children[node_idx:]
    _has_statement = False
    next_statement = None
    for sibling in siblings:
        if type(sibling.data).__name__ in STATEMENT_TYPE:
            _has_statement = True
            next_statement = sibling
            break
    if _has_statement:
        return next_statement
    elif not _has_statement and type(parent.data).__name__ in LOOP_TYPE:
        return parent

    return False


def seek_valid_fringe(node):
    children = list(node.children)
    exclude_nodes = ['BreakStatement', 'ContinueStatement']
    for child in children:
        if type(child.data).__name__  in exclude_nodes:
            continue
        else:
            if type(child.data).__name__ in STATEMENT_TYPE:
                return child
    return node