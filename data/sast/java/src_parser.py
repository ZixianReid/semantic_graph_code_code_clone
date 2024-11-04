
from util.setting import log
from data.sast.java.ast_parser import ASTParser
from data.sast.java.query_pattern import JAVA_QUERY
from data.sast.java.ast_builder import build_func_sast
from data.sast.java.fun_unit import FunUnit


exclude_type = [",","{",";","}",")","(",'"',"'","`",""," ","[]","[","]",":",".","''","'.'","b", "\\", "'['", "']","''", "comment", "@", "?"]

def extract_filename(file_path: str) -> str:
    """Extract file name excluding extension from file path.

    attributes:
        file_path -- the path of current file.
    
    returns:
        file_name -- the name of current file.
    """
    file_name = ''
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]

    if file_name == '':
        log.debug('Can not extract file name for path: {}' .format(file_path))
        exit(-1)
    
    return file_name


def java_parser(file_path: str) -> list:
    """ Parse Java source code file & extract function unit

    attributes:
        file_path -- the path of Java source file.
    
    returns:
        func_list -- list including all function in current file.
    """
    func_list = []
    parser = ASTParser('java')
    with open(file_path, 'rb') as f:
        serial_code = f.read()
        code_ast = parser.parse(serial_code)
    
    root_node = code_ast.root_node

    # print(root_node.sexp())

    # obtain file name
    file_name = extract_filename(file_path)

    query = JAVA_QUERY()

    # query methods
    _methods = query.class_method_query().captures(root_node)

    if len(_methods) !=1:
        print(file_name)
        log.error('more than one function! exit')
    else:
        _method = _methods[0]
        _m_name_tmp = query.method_declaration_query().captures(_method[0])
        _m_name = serial_code[_m_name_tmp[0][0].start_byte:_m_name_tmp[0][0].end_byte].decode('utf8')
        sast = build_func_sast(file_name, _m_name, _method[0], serial_code, exclude_type)
        func = FunUnit(sast, file_name, _m_name)

    return func