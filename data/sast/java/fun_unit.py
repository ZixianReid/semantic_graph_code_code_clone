
from treelib import Tree
from util.setting import log
from util.data_structure import Stack
from typing import Tuple, List

class FunUnit():
    """Maintain data for each function (e.g., file_name, func_name, parameter_num)

    attributes:
        sast -- instance of simplified Abstract Syntax Tree\\
        file_name -- path of the file including current function\\
        func_name -- name of current function\\
        parameter_type -- parameter type of current function\\
        parameter_name -- parameter name of current function \\
        import_header -- header used by this function\\
        field_params -- class field parameters for data dependency graph
        
    """
    
    def __init__(self, sast: Tree, file_name: str = None, func_name: str = None, parameter_type: list = [], parameter_name: list = [], import_header: list = [], field_params: list = []) -> None:
        """ Constructor of FunUnit class
        """
        if file_name == None or func_name == None:
            log.debug('FunUnit lacks essential params. file_name: {}, func_name: {}' .format(file_name, func_name))
            exit(-1)
        self.sast = sast
        self.file_name = file_name
        self.func_name = func_name
    

    def gen_type_sequence(self) -> list:
        """Depth-first search for generating type sequence.
        """
        sequence = list()
        root = self.sast.root

        stack = Stack()
        stack.push(root)

        while not stack.is_empty():
            current_node = stack.pop()
            _node_data = self.sast.get_node(current_node).data
            current_node_type = _node_data.node_type
            current_node_type = current_node_type.strip('\n').replace(',', ' ')
            sequence.append(current_node_type)
            children = self.sast.children(current_node)
            children.reverse()
            for child in children:
                stack.push(child.identifier)
        
        return sequence


    def gen_token_sequence(self) -> list:
        sequence = list()
        root = self.sast.root

        stack = Stack()
        stack.push(root)

        while not stack.is_empty():
            current_node = stack.pop()
            _node_data = self.sast.get_node(current_node).data
            current_node_token = _node_data.node_token
            sequence.append(current_node_token)
            children = self.sast.children(current_node)
            children.reverse()
            for child in children:
                stack.push(child.identifier)
        
        return sequence

    def has_type(self, type: str) -> bool:
        """Determine whether the sast contains specifc type.
        """
        type_sequence = self.gen_type_sequence()
        if type in type_sequence:
            return True
        
        return False

    def gen_identifer_token_sequence(self) -> dict:
        sequence = dict()
        root = self.sast.root

        stack = Stack()
        stack.push(root)

        while not stack.is_empty():
            current_node = stack.pop()
            _node_data = self.sast.get_node(current_node).data
            current_node_type = _node_data.node_type
            current_node_type = current_node_type.strip('\n').replace(',', ' ')
            current_node_identifier = current_node

            sequence[current_node_identifier] = current_node_type
            
            children = self.sast.children(current_node)
            children.reverse()
            for child in children:
                stack.push(child.identifier)
        return sequence


    def gen_ast_edge(self) -> List[List[str]]:
        """Generate ast edge list from ast list.
        """
        src = []
        tgt = []
        edge_type = []
        root = self.sast.root
        stack = Stack()
        stack.push(root)

        while not stack.is_empty():
            current_node = stack.pop()
            children = self.sast.children(current_node)
            children.reverse()
            for child in children:
                src.append(current_node)
                tgt.append(child.identifier)
                edge_type.append('ast')
                stack.push(child.identifier)
        
        return [src, tgt, edge_type]

