
from treelib import Tree
from util.setting import log
from util.data_structure import Stack

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
            sequence.append(current_node_type)
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