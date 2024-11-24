
from data.graph_builder.code_graph import EDGE_DICT, Queue
from util.setting import log
import networkx as nx

class DDGNode():
    """Node structure used for Data Dependency Graph construction. In general, we traverse the ast-based control flow graph, find each control flow node's def-use information.
    """
    def __init__(self, node_key: str = None, node_type: str = None, defs: list = [], uses : list = [], unknow: list = []) -> None:
        if node_key == None:
            logger.error('DDGNode initialization need node type, exit.')
            exit(-1)
        self.node_key = node_key
        self.node_type = node_type
        self.defs = defs
        self.uses = uses
        self.unknown = unknow
    
    def get_key(self) -> str:
        return self.node_key
        
    def get_defs(self) -> list:
        return self.defs
    
    def get_uses(self) -> list:
        return self.uses
    
    def get_unknown(self) -> list:
        return self.unknown
    
    def print_defs_uses(self) -> None:
        print('***DDGNode info***')
        print('Node Key: {}\nNode Type: {}\nNode Defs: {}\nNode Uses: {}\nNode Unknown: {}' .format(self.node_key, self.node_type, ','.join(self.defs), ','.join(self.uses), ','.join(self.unknown)))


def transfer_from_anynode_2_diagraph(root, src, tgt, edgetype):
    ## Transfer the edge information from anynode to diagraph to build the ddg graph following the same structure as Tailor shows.
    G = nx.MultiDiGraph()
    G.add_node(root.id, token=root.token, data=root.data, is_statement=root.is_statement)
    for node in root.descendants:
        G.add_node(
            node.id,
            token=node.token,
            data=node.data,
            is_statement=node.is_statement
        )
    for s, t, e_type in zip(src, tgt, edgetype):
        G.add_edge(s, t, edge_type=e_type)

    return G

def transfer_from_diagraph_2_anynode(G):
    src = []
    tgt = []
    edgetype = []
    for s, t, attrs in G.edges(data=True):
        src.append(s)
        tgt.append(t)
        edgetype.append(attrs.get("edge_type", None))
    return src, tgt, edgetype

def get_dfg_edge(node, src, tgt, edgetype):
    cfg_cpg = transfer_from_anynode_2_diagraph(node, src, tgt, edgetype)


    def_use_chain = gen_def_use_chain(cfg_cpg, node.id)
    for key, value in def_use_chain.items():
        if value.uses == []:
            continue
        for use_var in value.uses:
            back_tracking(cfg_cpg, key, def_use_chain, use_var)

    src, tgt, edgetype = transfer_from_diagraph_2_anynode(cfg_cpg)
    return node, src, tgt, edgetype


def gen_def_use_chain(cfg_cpg, node):
    ddg_chain = dict()
    check_exist = list()
    queue = Queue()
    queue.push(node)
    check_exist.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        ddg_node = merge_def_use(cfg_cpg, current_node)
        if ddg_node.node_key in ddg_chain:
            log.error('Appear duplicated Dict key, exit.')
            exit(-1)
        ddg_chain[ddg_node.node_key] = ddg_node
        node_successors = list(cfg_cpg.successors(current_node))
        for _succ in node_successors:
            if cfg_cpg.nodes[_succ]['is_statement'] and _succ not in check_exist and check_edge(cfg_cpg, current_node, _succ):
                queue.push(_succ)
                check_exist.append(_succ)
    return ddg_chain


def back_tracking(cfg_cpg, node, def_use_chain, use_var):
    visited = list()
    queue = Queue()
    if node == None:
        return True

    queue.push(node)
    while not queue.is_empty():
        current_node = queue.pop()
        visited.append(current_node)
        predecessors = list(cfg_cpg.predecessors(current_node))
        for _pred in predecessors:
            is_statement = cfg_cpg.nodes[_pred]['is_statement']
            if is_statement and _pred not in visited and check_edge(cfg_cpg, _pred, current_node):
                if _pred not in def_use_chain:
                    log.error(f"Parent node {_pred} not in def_use_chain")
                    exit(-1)
                ddg_predecessor = def_use_chain[_pred]
                if has_dd_rel(use_var, ddg_predecessor):
                    cfg_cpg.add_edge(ddg_predecessor.node_key, node, edge_type=[EDGE_DICT['dfg_edge']])
                else:
                    queue.push(_pred)
def has_dd_rel(use_var: str = None, def_node: DDGNode = None) -> bool:
    """Check whether use_var is defined by def_node.
    """

    if use_var == None or def_node == None:
        logger.error('Use variable or Def Node is None, Exit.')
        exit(-1)
    defs = def_node.get_defs()
    _dd_rel = False
    if use_var in defs:
        _dd_rel = True
    return _dd_rel


def check_edge(cfg_cpg, start, end):
    
    if cfg_cpg.has_edge(start, end):
        for key, edge_data in cfg_cpg[start][end].items():
            edge_type = edge_data.get('edge_type', None)
            if edge_type[0][0] == EDGE_DICT['dfg_edge'][0] or edge_type[0][0] == EDGE_DICT['cfg_edge'][0]:
                return True
    return False
        

def merge_def_use(cfg_cpg, node):
    """Parse one statement node, and traverse its all child entities (non cfg) to collect its def use information and Merge them together.
    """
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _succ)
        defs += _s_def
        uses += _s_use
        unknown += _s_unknown

    tmp_defs = list(set(defs))
    tmp_uses = list(set(uses))
    tmp_unknown = list(set(unknown))

    tmp_defs.sort(key=defs.index)
    tmp_uses.sort(key=uses.index)
    tmp_unknown.sort(key=unknown.index)

    node_type = type(cfg_cpg.nodes[node]['data']).__name__
    if node_type == 'ReturnStatement':
       tmp_uses = tmp_uses + tmp_unknown
       tmp_unknown = []
    
    return DDGNode(node, node_type, tmp_defs, tmp_uses, tmp_unknown)


def extract_def_use(cfg_cpg, node):
    node_type = type(cfg_cpg.nodes[node]['data']).__name__
    if node_type == "BinaryOperation":
        defs, uses, unknown = ddg_binaryoperation(cfg_cpg, node)
    elif node_type == "MemberReference":
        defs, uses, unknown = ddg_memberreference(cfg_cpg, node)
    elif node_type == "Assignment":
        defs, uses, unknown = ddg_assignment(cfg_cpg, node)
    elif node_type == "VariableDeclarator":
        defs, uses, unknown = ddg_variabledeclarator(cfg_cpg, node)
    elif node_type == "ClassCreator":
        defs, uses, unknown = ddg_classcreator(cfg_cpg, node)
    elif node_type == "ArraySelector":
        defs, uses, unknown = ddg_arrayselector(cfg_cpg, node)
    elif node_type == "MethodInvocation":
        defs, uses, unknown = ddg_methodinvocation(cfg_cpg, node)
    elif node_type == "FormalParameter":
        defs, uses, unknown = ddg_formalparameter(cfg_cpg, node)
    elif node_type == "Literal":
        defs, uses, unknown = ddg_literal(cfg_cpg, node)
    elif node_type == "TernaryExpression":
        defs, uses, unknown = ddg_ternaryexpression(cfg_cpg, node)
    elif node_type == "Cast":
        defs, uses, unknown = ddg_cast(cfg_cpg, node)
    elif node_type == 'str':
        defs, uses, unknown = ddg_string(cfg_cpg, node)
    else:
        defs, uses, unknown = ddg_deeper(cfg_cpg, node)
    return [defs, uses, unknown]

def ddg_deeper(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) == 0:
        return [defs, uses, unknown]
    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, _succ)
        defs = defs + _f_def
        uses = uses + _f_use
        unknown = unknown + _f_unknown
    return [defs, uses, unknown]


def ddg_cast(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, _succ)
        uses = uses + _f_def + _f_use + _f_unknown
    return [defs, uses, unknown]


def ddg_ternaryexpression(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        log.error(f"TernaryExpression node has {len(node_successors)} children, expected 3")
        exit(-1)

    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, _succ)
        uses = uses + _f_def + _f_use + _f_unknown

    return [defs, uses, unknown]


def ddg_string(cfg_cpg,node):
    defs, uses = [], []
    unknown = [cfg_cpg.nodes[node]['token']]
    return [defs, uses, unknown]


def ddg_formalparameter(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(cfg_cpg, _succ)
        defs = defs + _s_def + _s_unknown

    return [defs, uses, unknown]



def ddg_methodinvocation(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, _succ)
        uses = uses + _f_def + _f_use + _f_unknown
    return [defs, uses, unknown]


def ddg_literal(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 1:
        log.error(f"Literal node has {len(node_successors)} children, expected 1")
        exit(-1)
    _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
    uses = uses + _f_def + _f_use + _f_unknown
    return [defs, uses, unknown]

def ddg_arrayselector(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])
    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])
    uses = _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown
    return [defs, uses, unknown]

def ddg_classcreator(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    for _succ in node_successors:
        if cfg_cpg.nodes[_succ]['is_statement']:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(_succ)
        uses = uses + _s_def + _s_use + _s_unknown
    return [defs, uses, unknown]


def ddg_variabledeclarator(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) not in [1, 2]:
        log.error(f"VariableDeclarator node has {len(node_successors)} children, expected 2")
        exit(-1)
    
    if len(node_successors) == 1:
        _f_defs, _f_uses, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
        defs = defs + _f_defs + _f_unknown
    else:
        _f_defs, _f_uses, _f_unknown = extract_def_use(cfg_cpg, node_successors[-1])
        _t_defs, _t_uses, _t_unknown = extract_def_use(cfg_cpg, node_successors[0])
        defs = defs + _f_defs + _f_unknown
        uses = uses + _f_uses + _t_defs + _t_uses + _t_unknown
    return [defs, uses, unknown]


def ddg_assignment(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))
    if len(node_successors) != 3:
        log.error(f"Assignment node has {len(node_successors)} children, expected 3")
        exit(-1)

    _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
    defs = defs + _f_def + _f_unknown 
    uses = uses + _f_use

    _t_def, _t_use, _t_unknown = extract_def_use(cfg_cpg, node_successors[1])
    uses = uses + _t_use + _t_unknown + _t_def

    return [defs, uses, unknown]


def ddg_binaryoperation(cfg_cpg, node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    if len(node_successors) != 3:
        log.error(f"BinaryOperation node has {len(node_successors)} children, expected 3")
        exit(-1)
    
    _f_def, _f_use, _f_unknown = extract_def_use(cfg_cpg, node_successors[1])
    uses = uses + _f_def + _f_use + _f_unknown

    _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[0])
    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]


def ddg_memberreference(cfg_cpg,node):
    defs, uses, unknown = [], [], []
    node_successors = list(cfg_cpg.successors(node))

    if len(node_successors) == 1:
        _f_defs, _f_uses, _f_unknown = extract_def_use(cfg_cpg, node_successors[0])
        uses = uses + _f_defs + _f_uses + _f_unknown
    else:
        
        _l_def, _l_use, _l_unknown = extract_def_use(cfg_cpg, node_successors[0])
        _r_def, _r_use, _r_unknown = extract_def_use(cfg_cpg, node_successors[-1])
    
        uses = _l_def + _l_use + _l_unknown + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]


def combine_fields(left_v: list = None, right_v: list = None) -> list:
    """Combine two variable list and generate new def use variable.
    """
    res_v = list()

    if left_v == [] or res_v == []:
        return res_v

    longest_pre = left_v[0]
    for l_v in left_v:
        if len(l_v) > len(longest_pre):
            longest_pre = l_v
    for r_v in right_v:
        res_v.append(longest_pre + '.' + r_v)

    return res_v
