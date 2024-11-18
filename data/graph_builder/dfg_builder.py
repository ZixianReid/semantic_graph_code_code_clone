
from data.graph_builder.code_graph import EDGE_DICT, Queue
from util.setting import log

def get_dfg_edge(node, src, tgt, edgetype):
    pass





def gen_def_use_chain(node):
    ddg_chain = dict()
    check_exist = list()
    queue = Queue()
    queue.push(node)
    check_exist.append(node)

    while not queue.is_empty():
        current_node = queue.pop()
        ddg_node = merge_def_use(current_node)



def merge_def_use(node):
    """Parse one statement node, and traverse its all child entities (non cfg) to collect its def use information and Merge them together.
    """
    defs, uses, unknown = [], [], []
    node_children = list(node.children)

    for child in node_children:
        if child.is_statement:
            continue
        _s_def, _s_use, _s_unknown = extract_def_use(child)

    


def extract_def_use(node):
    node_type = type(node.data).__name__
    if node_type == "BinaryOperation":
        defs, uses, unknown = ddg_binaryoperation(node)
    elif node_type == "MemberReference":
        defs, uses, unknown = ddg_memberreference(node)

    elif node_type == "Assignment":
        defs, uses, unknown = ddg_assignment(node)

    elif node_type == "VariableDeclarator":
        defs, uses, unknown = ddg_variabledeclarator(node)

    else:
         defs, uses, unknown = ddg_deeper(node)


def ddg_deeper(node):
    defs, uses, unknown = [], [], []
    node_children = list(node.children)
    if len(node_children) == 0:
        return [defs, uses, unknown]
    for child in node_children:
        if node.is_statement:
            continue
        _f_def, _f_use, _f_unknown = extract_def_use(child)
        defs = defs + _f_def
        uses = uses + _f_use
        unknown = unknown + _f_unknown
    return [defs, uses, unknown]

def ddg_variabledeclarator(node):
    pass

def ddg_assignment(node):
    defs, uses, unknown = [], [], []
    children = list(node.children)
    if len(children) != 3:
        log.error(f"Assignment node has {len(children)} children, expected 3")
        exit(-1)

    _f_def, _f_use, _f_unknown = extract_def_use(children[0])
    defs = defs + _f_def + _f_unknown 
    uses = uses + _f_use

    _t_def, _t_use, _t_unknown = extract_def_use(children[1])
    uses = uses + _t_use + _t_unknown + _t_def

    return [defs, uses, unknown]


def ddg_binaryoperation(node):
    defs, uses, unknown = [], [], []
    children = list(node.children)

    if len(children) != 3:
        log.error(f"BinaryOperation node has {len(children)} children, expected 3")
        exit(-1)
    
    _f_def, _f_use, _f_unknown = extract_def_use(children[1])
    uses = uses + _f_def + _f_use + _f_unknown

    _r_def, _r_use, _r_unknown = extract_def_use(children[-1])
    uses = uses + _r_def + _r_use + _r_unknown

    return [defs, uses, unknown]


def ddg_memberreference(node):
    defs, uses, unknown = [], [], []
    children = list(node.children)
    if not len(children) == 2:
        log.error(f"MemberReference node has {len(children)} children, expected 2")
        exit(-1)
    
    _l_def, _l_use, _l_unknown = extract_def_use(children[0])
    _r_def, _r_use, _r_unknown = extract_def_use(children[-1])
    
    defs = _r_def + _r_use + _r_unknown
    uses = _l_def + _l_use + _l_unknown
    combinations = combine_fields(uses, defs)
    defs = combinations
    unknown = []

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
